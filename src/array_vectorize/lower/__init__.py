# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""AST -> IR lowering.

The package assembles ``_Lowerer`` from one concern per layer — state
and environment machinery (``env``), argument sanitization
(``sanitize``), expressions (``expressions``), statements
(``statements``), loops (``loops``), and user helpers (``helpers``) —
with each layer extending the previous one, so all state lives on the
single base class. This module holds the final assembly (the statement
dispatcher) and the ``lower_function`` entry point. Lowering handles
straight-line code, if/elif/else with early returns, constant-trip
loops with loop-carried phis (design D6) and calls to other vectorizable
scalar functions (design D7).
"""

from __future__ import annotations

import ast
from typing import Any

from array_vectorize.errors import VectorizationError
from array_vectorize.frontend.info import FunctionInfo
from array_vectorize.ir import generated_name
from array_vectorize.ir import Node
from array_vectorize.ir import Program
from array_vectorize.lower.helpers import _HelperLowerer
from array_vectorize.lower.types import HelperVectorizer
from array_vectorize.lower.types import LoweredFunction
from array_vectorize.runtime.registry import RUNTIME_HELPERS

__all__ = ["HelperVectorizer", "LoweredFunction", "lower_function"]


class _Lowerer(_HelperLowerer):
    def lower_stmts(self, stmts: list[ast.stmt], *, loop_body: bool = False) -> Node | None:
        """Lower a statement list; returns the merged result if a return was hit.

        With ``loop_body=True`` (a loop body), loop-carried variables are
        synchronized back to their loop names after every statement, so the
        real generated loop feeds the right values across iterations.
        """
        for stmt in stmts:
            if isinstance(stmt, ast.Assign | ast.AugAssign):
                self.lower_assign(stmt)
            elif isinstance(stmt, ast.Return):
                if self.loop_depth > 0:
                    raise self.error(
                        stmt, "returns inside loop bodies are not supported (no early loop exit)"
                    )
                assert stmt.value is not None  # validator rejects bare returns
                return self.fold_returns(self.lower_expr(stmt.value))
            elif isinstance(stmt, ast.If):
                merged = self.lower_if(stmt)
                if merged is not None:
                    return merged
            elif isinstance(stmt, ast.For):
                self.lower_for(stmt)
            elif isinstance(stmt, ast.Expr | ast.Pass):
                if loop_body:
                    self._sync_loop_carried()
                continue  # docstring / pass
            else:
                raise self.error(stmt, f"{type(stmt).__name__} statements are not supported")
            if loop_body:
                self._sync_loop_carried()
        return None


def lower_function(
    info: FunctionInfo,
    helper_vectorizer: HelperVectorizer | None = None,
) -> LoweredFunction:
    """Lower a validated FunctionInfo to IR."""
    lowerer = _Lowerer(info, helper_vectorizer=helper_vectorizer)
    # bind parameter names in order (first bind keeps the name, mangling reserved)
    # the namespace variable: `xp` unless the user uses that name; allocated
    # through the shared SSA env (and reserved) so rebinding a user `xp`
    # can never collide with it
    ns = "xp" if lowerer.ssa.is_free("xp") else lowerer.ssa.bind("xp", force_suffix=True)
    lowerer.ssa.reserve(ns)
    lowerer.ns_var = ns
    # runtime dtype promotion helpers: allocated through the shared SSA env
    # so a user parameter named like a helper cannot shadow it (bare-name
    # calls would resolve to the parameter instead of the global)
    for attr, key in (("minmax_name", "vec_minmax"), ("arith_name", "vec_arith")):
        base, fn = RUNTIME_HELPERS[key]
        helper_name = (
            base if lowerer.ssa.is_free(base) else lowerer.ssa.bind(base, force_suffix=True)
        )
        lowerer.ssa.reserve(helper_name)
        setattr(lowerer, attr, helper_name)
        lowerer.helpers.append((helper_name, fn))
    param_names = [lowerer.ssa.bind(p.name) for p in info.params]
    # after params take their names, block the caller's own generated name so
    # a same-named helper must pick a different name (emitted-only: user
    # parameters keep their names and keyword calls keep working)
    lowerer.ssa.emitted.add(generated_name(info.name))
    for param, emitted in zip(info.params, param_names, strict=True):
        lowerer.definite[param.name] = emitted
        # parameters can arrive as raw scalars (omitted defaults); operators
        # promote them, xp.* call sites wrap them (see _sanitize_xp_arg)
        lowerer.kinds.mark_scalar(emitted)
    hidden: list[tuple[str, Any]] = []
    for var, array in info.closure_arrays.items():
        emitted = lowerer.ssa.bind(var)
        lowerer.definite[var] = emitted
        hidden.append((emitted, array))
    # the hidden kw-only namespace parameter for pinned mode
    # (vectorize(namespace=...)): allocated AFTER user params and
    # closure-hidden params take their names, so a user parameter named
    # ``_namespace`` keeps its name and ours is mangled instead; reserved
    # like ``xp`` so body bindings can never collide with it. Not a
    # program parameter: the namespace is only consumed by the codegen-
    # level ``xp`` binding, never by body IR.
    ns_param = (
        "_namespace"
        if lowerer.ssa.is_free("_namespace")
        else lowerer.ssa.bind("_namespace", force_suffix=True)
    )
    lowerer.ssa.reserve(ns_param)

    tree = info.tree
    maybe_result: Node | None
    if isinstance(tree, ast.Lambda):
        maybe_result = lowerer.lower_expr(tree.body)
    else:
        maybe_result = lowerer.lower_stmts(tree.body)
    if maybe_result is None:
        raise VectorizationError(
            f"cannot vectorize {info.name!r}: not all control-flow paths return a value"
        )
    result: Node = maybe_result

    program = Program(
        params=tuple(param_names) + tuple(name for name, _ in hidden),
        bindings=tuple(lowerer.bindings),
        result=result,
    )
    return LoweredFunction(
        program=program,
        name=info.name,
        params=info.params,
        param_names=param_names,
        hidden_params=hidden,
        source=info.source,
        docstring=info.docstring,
        helpers=lowerer.helpers,
        namespace_var=ns,
        namespace_param=ns_param,
        emitted_names=frozenset(lowerer.ssa.emitted),
    )
