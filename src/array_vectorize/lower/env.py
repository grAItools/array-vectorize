# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Lowering state and the shared environment machinery.

``_LowererBase`` owns every piece of mutable lowering state — SSA name
allocation, definite/maybe name tracking, deferred early returns, loop
frames, and the kind/literal/scalar bookkeeping — and provides the
utilities every layer builds on: error reporting, literal construction,
boolean coercion, name resolution, and the branch-merge machinery
(environment merges, maybe-unbound marking, return folding).
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Any, TYPE_CHECKING

from array_vectorize import errors
from array_vectorize import ir
from array_vectorize.frontend import info
from array_vectorize.lower import kinds
from array_vectorize.lower import types
from array_vectorize.runtime import registry

__all__ = ["_LowererBase"]


class _LowererBase:
    if TYPE_CHECKING:
        # implemented by the assembled _Lowerer / its mixins (mutual
        # recursion: the statement dispatcher drives loop lowering, loop
        # bodies are lowered through the dispatcher, and the sanitizer /
        # expression / helper layers provide kind inference, maybe-bool
        # detection, and helper dispatch from above this base). Type-only
        # declarations — never executed; the real methods on the assembled
        # class are the ones that run.
        def lower_stmts(
            self, stmts: list[ast.stmt], *, loop_body: bool = False
        ) -> ir.Node | None: ...
        def _numeric_kind(self, node: ir.Node) -> ir.Kind | None: ...
        def _maybe_bool_result(self, node: ir.Node) -> bool: ...
        def _call_helper(self, node: ast.Call, name: str, args: list[ir.Node]) -> ir.Node: ...

    def __init__(
        self,
        info: info.FunctionInfo,
        helper_vectorizer: types.HelperVectorizer | None = None,
    ) -> None:
        self.info = info
        self.helper_vectorizer = helper_vectorizer
        self.helpers: list[tuple[str, Any]] = []
        self._helper_names: dict[Callable[..., Any], str] = {}
        self.ssa = ir.SSAEnv(info.user_names)
        self.bindings: list[ir.Stmt] = []
        #: definite locals: var -> current emitted name
        self.definite: dict[str, str] = {}
        #: vars that may be unbound on the current path (assigned in one branch only)
        self.maybe: set[str] = set()
        #: pending early returns: (condition, value, push_depth) in divergence
        #: order; folds nest earlier entries outermost
        self.deferred: list[tuple[ir.Node, ir.Node, int]] = []
        #: nesting depth inside branches (branch-local bindings always suffix)
        self.branch_depth = 0
        #: nesting depth inside loop bodies (returns are rejected there)
        self.loop_depth = 0
        #: source names of active loop variables (assignment to them rejected)
        self.loop_vars: set[str] = set()
        #: stack of ({var: loop-carried emitted name}, entry branch depth) per loop
        self.active_carried: list[tuple[dict[str, str], int]] = []
        #: kind/literal/scalar/bool bookkeeping for emitted names, plus the
        #: per-loop carried-assign kind frames (see kinds.Kinds)
        self.kinds = kinds.Kinds()
        #: the generated namespace variable ('xp' unless taken); set by
        #: lower_function before lowering starts
        self.ns_var: str = "xp"
        #: allocated names of the runtime promotion/selection helpers (set
        #: by lower_function; collision-free against user names)
        self.minmax_name = registry.RUNTIME_HELPERS["vec_minmax"][0]
        self.arith_name = registry.RUNTIME_HELPERS["vec_arith"][0]
        self.assigned_names: set[str] = {
            t.id
            for stmt in ast.walk(info.tree)
            if isinstance(stmt, ast.Assign | ast.AugAssign | ast.For)
            for t in (
                stmt.targets
                if isinstance(stmt, ast.Assign)
                else [stmt.target]
                if isinstance(stmt, ast.AugAssign | ast.For)
                else []
            )
            if isinstance(t, ast.Name)
        }

    # ------------------------------------------------------------- utilities

    def error(self, node: ast.AST, message: str) -> errors.VectorizationError:
        lineno = getattr(node, "lineno", 0)
        return errors.VectorizationError(
            f"cannot vectorize {self.info.name!r}: {message} (line {lineno})"
        )

    def _literal(self, value: Any) -> ir.Literal:
        if isinstance(value, bool):
            return ir.Literal(value, "bool")
        if isinstance(value, int):
            return ir.Literal(int(value), "int")
        return ir.Literal(float(value), "float")

    def _coerce_bool(self, node: ir.Node) -> ir.Node:
        """Coerce a possibly-numeric condition to bool via != 0 (design D2/D4)."""
        if ir.is_bool(node):
            return node
        return ir.Compare("ne", node, ir.Literal(0, "int"))

    # ---------------------------------------------------------- name access

    def load_name(self, node: ast.Name) -> ir.Node:
        var = node.id
        if var in self.definite:
            return ir.Ref(self.definite[var])
        if var in self.maybe:
            raise self.error(node, f"variable {var!r} may be unbound (assigned on some paths only)")
        if var in self.assigned_names:
            raise self.error(node, f"local variable {var!r} referenced before assignment")
        info = self.info
        if var in info.closure_scalars:
            return self._literal(info.closure_scalars[var])
        if var in info.closure_arrays:
            return ir.Ref(self.definite[var])  # hidden param, bound at intake
        if var in info.math_funcs or var in info.user_funcs or var in info.math_modules:
            raise self.error(node, f"cannot reference {var!r} without calling it")
        raise self.error(node, f"name {var!r} is not supported in vectorized code")

    def _mark_then_only_unbound(
        self,
        pre: dict[str, str],
        returning: dict[str, str],
        continuing: dict[str, str],
    ) -> None:
        """Vars bound only on the returning path are maybe-unbound afterwards."""
        for var in returning:
            if var not in continuing and var not in pre:
                self.maybe.add(var)
                self.definite.pop(var, None)

    def _merge_envs(
        self,
        cond: ir.Node,
        pre: dict[str, str],
        then_def: dict[str, str],
        else_def: dict[str, str],
        then_maybe: set[str],
        else_maybe: set[str],
    ) -> None:
        for var in dict.fromkeys([*then_def, *else_def]):  # deterministic order
            nt, ne = then_def.get(var), else_def.get(var)
            if nt == ne:
                continue
            if nt is not None and ne is not None:
                name = self.ssa.bind(var)
                merged_expr: ir.Node = ir.Where(cond, ir.Ref(nt), ir.Ref(ne))
                self.bindings.append(ir.Binding(name, merged_expr))
                self.definite[var] = name
                kt, ke = self.kinds.kind(nt), self.kinds.kind(ne)
                self.kinds.set_kind(name, kt if kt == ke else None)
                if self._maybe_bool_result(merged_expr):
                    self.kinds.mark_maybe_bool(name)
            else:
                # bound on one path only (and not before the if): maybe-unbound
                self.definite.pop(var, None)
                self.maybe.add(var)
        for var in then_maybe | else_maybe:
            self.definite.pop(var, None)
            self.maybe.add(var)
        for var in list(self.maybe):
            if var in then_def and var in else_def and then_def[var] == else_def[var]:
                self.maybe.discard(var)
                self.definite[var] = then_def[var]

    def fold_returns(self, value: ir.Node) -> ir.Node:
        """Fold pending early returns around ``value``.

        Entries pushed at the current branch depth or deeper belong to this
        block: they are applied here (list order is divergence order, so the
        fold nests earlier returns outermost). Entries pushed at shallower
        depths belong to enclosing blocks and are retained for their folds;
        each branch copy carries them, and mixed ifs insert their own
        divergence before enclosed entries.
        """
        result = value
        retained: list[tuple[ir.Node, ir.Node, int]] = []
        for cond, early, depth in reversed(self.deferred):
            if depth < self.branch_depth:
                retained.append((cond, early, depth))
            else:
                result = ir.Where(cond, early, result)
        retained.reverse()
        self.deferred = retained
        return result
