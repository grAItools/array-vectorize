"""AST -> IR lowering (plan §5 translation rules, §6 IR, §7 semantics).

Handles straight-line code (M1), if/elif/else with early returns (M2),
constant-trip loops with loop-carried phis (M3, D6) and calls to other
vectorizable scalar functions (M3, D7).
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ._builtins import (
    BUILTIN_CASTS,
    BUILTIN_FOLDS,
    BUILTIN_UNARY,
    MATH_CONSTS,
    MATH_FUNCS,
    MATH_SPECIAL,
)
from ._errors import VectorizationError
from ._extract import FunctionInfo, Param
from ._ir import (
    Binding,
    BinOp,
    Call,
    Compare,
    DType,
    FuncCall,
    Literal,
    Logical,
    Loop,
    Node,
    Program,
    Ref,
    SSAEnv,
    Stmt,
    UnaryOp,
    Where,
    generated_name,
    is_bool,
)
from ._optimize import _children
from ._runtime import _ARITH_OPS, _vec_arith, _vec_minmax

__all__ = ["LoweredFunction", "lower_function"]

_BINOPS: dict[type[ast.operator], str] = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
    ast.Pow: "pow",
    ast.FloorDiv: "floordiv",
    ast.Mod: "mod",
    ast.BitAnd: "and",
    ast.BitOr: "or",
    ast.BitXor: "xor",
    ast.LShift: "lshift",
    ast.RShift: "rshift",
}
_CMPOPS: dict[type[ast.cmpop], str] = {
    ast.Eq: "eq",
    ast.NotEq: "ne",
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
}
_UNARY: dict[type[ast.unaryop], str] = {
    ast.USub: "neg",
    ast.UAdd: "pos",
    ast.Invert: "invert",
    ast.Not: "not",
}


@dataclass
class LoweredFunction:
    """Lowering output: IR plus everything codegen/runtime need."""

    program: Program
    name: str
    params: list[Param]
    param_names: list[str]  # emitted names, aligned with params
    hidden_params: list[tuple[str, Any]]  # (emitted name, array default)
    source: str  # original scalar source (embedded verbatim in the docstring)
    helpers: list[tuple[str, Any]]  # (emitted name, vectorized helper callable)
    namespace_var: str  # the generated code's ``xp`` (renamed on collision)
    emitted_names: frozenset[str]  # every binding/param name the lowering emitted


HelperVectorizer = Callable[[Callable[..., Any]], Callable[..., Any]]


class _Lowerer:
    def __init__(
        self,
        info: FunctionInfo,
        helper_vectorizer: HelperVectorizer | None = None,
    ) -> None:
        self.info = info
        self.helper_vectorizer = helper_vectorizer
        self.helpers: list[tuple[str, Any]] = []
        self._helper_names: dict[Callable[..., Any], str] = {}
        self.ssa = SSAEnv(info.user_names)
        self.bindings: list[Stmt] = []
        #: definite locals: var -> current emitted name
        self.definite: dict[str, str] = {}
        #: vars that may be unbound on the current path (assigned in one branch only)
        self.maybe: set[str] = set()
        #: pending early returns: (condition, value, push_depth) in divergence
        #: order; folds nest earlier entries outermost
        self.deferred: list[tuple[Node, Node, int]] = []
        #: nesting depth inside branches (branch-local bindings always suffix)
        self.branch_depth = 0
        #: nesting depth inside loop bodies (returns are rejected there)
        self.loop_depth = 0
        #: source names of active loop variables (assignment to them rejected)
        self.loop_vars: set[str] = set()
        #: stack of ({var: loop-carried emitted name}, entry branch depth) per loop
        self.active_carried: list[tuple[dict[str, str], int]] = []
        #: emitted name -> provable numeric kind ('int'/'float'/'bool' or
        #: None when unknown). Names are unique (SSA), so entries never go
        #: stale; this lets _numeric_kind see through Ref nodes.
        self.name_kinds: dict[str, str | None] = {}
        #: emitted name -> its Literal value, for bindings of plain literals
        #: (lets later uses substitute the value instead of emitting casts on
        #: runtime plain scalars, which would crash xp.astype/xp.sqrt)
        self.name_literals: dict[str, Literal] = {}
        #: per active loop: {loop-carried name -> kinds assigned in the body}.
        #: A carried variable's runtime kind is the union of its phi kind and
        #: every body assignment; mixed kinds need conservative handling.
        self._carried_assign_kinds: list[dict[str, set[str | None]]] = []
        #: emitted names whose runtime value may be a raw Python scalar:
        #: parameters (omitted defaults), loop variables (raw ints per
        #: iteration), literal bindings, and locals computed from them.
        #: Operators promote scalars fine; xp.* function arguments do not
        #: (strict backends reject plain scalars), so call sites wrap these.
        self._scalar_names: set[str] = set()
        #: the generated namespace variable ('xp' unless taken); set by
        #: lower_function before lowering starts
        self.ns_var: str = "xp"
        #: names whose value is a min/max result with unknown static kind:
        #: the runtime dtype may be boolean (parameter operands), so
        #: arithmetic must use the runtime-polymorphic intify (a no-op for
        #: numeric dtypes)
        self._maybe_bool_names: set[str] = set()
        #: allocated names of the runtime promotion/selection helpers (set
        #: by lower_function; collision-free against user names)
        self.minmax_name = "_vec_minmax"
        self.arith_name = "_vec_arith"
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

    def error(self, node: ast.AST, message: str) -> VectorizationError:
        lineno = getattr(node, "lineno", 0)
        return VectorizationError(f"cannot vectorize {self.info.name!r}: {message} (line {lineno})")

    def _literal(self, value: Any) -> Literal:
        if isinstance(value, bool):
            return Literal(value, "bool")
        if isinstance(value, int):
            return Literal(int(value), "int")
        return Literal(float(value), "float")

    def _coerce_bool(self, node: Node) -> Node:
        """Coerce a possibly-numeric condition to bool via != 0 (plan D2/D4)."""
        if is_bool(node):
            return node
        return Compare("ne", node, Literal(0, "int"))

    # ---------------------------------------------------------- name access

    def load_name(self, node: ast.Name) -> Node:
        var = node.id
        if var in self.definite:
            return Ref(self.definite[var])
        if var in self.maybe:
            raise self.error(node, f"variable {var!r} may be unbound (assigned on some paths only)")
        if var in self.assigned_names:
            raise self.error(node, f"local variable {var!r} referenced before assignment")
        info = self.info
        if var in info.closure_scalars:
            return self._literal(info.closure_scalars[var])
        if var in info.closure_arrays:
            return Ref(self.definite[var])  # hidden param, bound at intake
        if var in info.math_funcs or var in info.user_funcs or var in info.math_modules:
            raise self.error(node, f"cannot reference {var!r} without calling it")
        raise self.error(node, f"name {var!r} is not supported in vectorized code")

    # --------------------------------------------------------- statements

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

    def lower_assign(self, stmt: ast.Assign | ast.AugAssign) -> None:
        if isinstance(stmt, ast.Assign):
            target = stmt.targets[0]
            assert isinstance(target, ast.Name)  # validator guarantees
            value = self.lower_expr(stmt.value)
        else:
            target = stmt.target
            assert isinstance(target, ast.Name)
            value = self._lower_binop(stmt.op, self.load_name(target), self.lower_expr(stmt.value))
        var = target.id
        if var in self.loop_vars:
            raise self.error(stmt, f"cannot assign the loop variable {var!r} inside its loop")
        name: str | None = None
        if self.active_carried:
            carried, entry_depth = self.active_carried[-1]
            if var in carried and self.branch_depth == entry_depth:
                # loop-carried by the innermost active loop, assigned at the
                # loop body's own level: write the loop name directly.
                # (Inside nested branches, normal suffixing + merge + the
                # post-statement sync keep the loop name consistent.)
                name = carried[var]
        if name is None:
            name = self.ssa.bind(var, force_suffix=self.branch_depth > 0)
        self.bindings.append(Binding(name, value))
        self.definite[var] = name
        kind = self._numeric_kind(value)
        self.name_kinds[name] = kind
        if isinstance(value, Literal):
            self.name_literals[name] = value
        else:
            self.name_literals.pop(name, None)
        if self._possibly_scalar(value):
            self._scalar_names.add(name)
        else:
            self._scalar_names.discard(name)
        if self._maybe_bool_result(value):
            self._maybe_bool_names.add(name)
        else:
            self._maybe_bool_names.discard(name)
        if self.active_carried:
            for carried, _depth in self.active_carried:
                if var in carried:
                    for frame in self._carried_assign_kinds:
                        frame.setdefault(carried[var], set()).add(kind)
        self.maybe.discard(var)

    def lower_if(self, stmt: ast.If) -> Node | None:
        cond = self.lower_expr(stmt.test)
        if not is_bool(cond):
            raise self.error(
                stmt,
                "bare truthiness in an if condition is ambiguous per-lane; "
                "write an explicit comparison, e.g. 'if x != 0:'",
            )
        pre_definite = dict(self.definite)
        pre_maybe = set(self.maybe)
        pre_deferred = list(self.deferred)

        # THEN branch
        self.branch_depth += 1
        then_result = self.lower_stmts(stmt.body)
        self.branch_depth -= 1
        then_definite, then_maybe = dict(self.definite), set(self.maybe)
        then_deferred = self.deferred

        # ELSE branch: reset to pre-state (bindings stay; both are emitted)
        self.definite, self.maybe = dict(pre_definite), set(pre_maybe)
        self.deferred = list(pre_deferred)
        self.branch_depth += 1
        else_result = self.lower_stmts(stmt.orelse)
        self.branch_depth -= 1
        else_definite, else_maybe = dict(self.definite), set(self.maybe)
        else_deferred = self.deferred
        pre_len = len(pre_deferred)

        if then_result is not None and else_result is not None:
            # Both branches return: this if terminates the block. Branch folds
            # applied their own entries; re-fold the merged value to apply
            # entries pending at this block's depth (they fire earlier in
            # program order, so they wrap the merged where from outside).
            self.deferred = list(pre_deferred)
            merged: Node = Where(cond, then_result, else_result)
            return self.fold_returns(merged)

        if then_result is not None:
            # then returns, else falls through: continue on the else path.
            # The if's own divergence encloses the else branch's pending
            # entries, so it is inserted before them (but after entries
            # pending from before this if).
            self.definite, self.maybe = else_definite, else_maybe
            self.deferred = [
                *else_deferred[:pre_len],
                (cond, then_result, self.branch_depth),
                *else_deferred[pre_len:],
            ]
            self._mark_then_only_unbound(pre_definite, then_definite, else_definite)
            return None
        if else_result is not None:
            # else returns, then falls through: continue on the then path.
            self.definite, self.maybe = then_definite, then_maybe
            not_cond = UnaryOp("not", cond)
            self.deferred = [
                *then_deferred[:pre_len],
                (not_cond, else_result, self.branch_depth),
                *then_deferred[pre_len:],
            ]
            self._mark_then_only_unbound(pre_definite, else_definite, then_definite)
            return None

        # both fall through: merge environments with where-selects
        self._merge_envs(cond, pre_definite, then_definite, else_definite, then_maybe, else_maybe)
        # qualify branch-pending returns with their branch conditions
        not_cond = UnaryOp("not", cond)
        qualified: list[tuple[Node, Node, int]] = []
        for c, v, p in then_deferred[pre_len:]:
            qualified.append((Logical("and", (cond, c)), v, p))
        for c, v, p in else_deferred[pre_len:]:
            qualified.append((Logical("and", (not_cond, c)), v, p))
        self.deferred = [*pre_deferred, *qualified]
        return None

    # ------------------------------------------------------------- loops

    def _const_int(self, node: ast.expr, what: str) -> Node:
        from ._optimize import _const_fold_expr

        folded = _const_fold_expr(self.lower_expr(node))
        if (
            isinstance(folded, Literal)
            and folded.kind == "int"
            and -(2**63) <= folded.value < 2**63
        ):
            return folded
        raise self.error(node, f"{what} must be constant ints known at generation time (plan D6)")

    def _loop_bounds(self, call: ast.Call) -> tuple[Node, Node, Node]:
        args = list(call.args)
        bounds: list[Node] = [self._const_int(a, "loop bound") for a in args]
        if len(bounds) == 1:
            start_n: Node = Literal(0, "int")
            stop_n: Node = bounds[0]
            step_n: Node = Literal(1, "int")
        elif len(bounds) == 2:
            start_n, stop_n = bounds
            step_n = Literal(1, "int")
        else:
            start_n, stop_n, step_n = bounds
        assert isinstance(step_n, Literal)
        if step_n.value == 0:
            raise self.error(call, "range step cannot be zero")
        return start_n, stop_n, step_n

    def _body_bound_names(self, stmts: list[ast.stmt], *, include_for_targets: bool) -> set[str]:
        """Names bound in a statement list.

        With ``include_for_targets``: also the targets of NESTED ``for``
        statements - Python loop targets bind in the enclosing scope, so a
        nested loop rebinding an outer variable is a loop-carried dependency
        of the outer loop. Without: only ``=``/augmented assignments (used
        for the cannot-assign-the-loop-variable check).
        """
        bound: set[str] = set()
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        bound.add(t.id)
            elif isinstance(stmt, ast.AugAssign):
                if isinstance(stmt.target, ast.Name):
                    bound.add(stmt.target.id)
            elif isinstance(stmt, ast.If):
                bound |= self._body_bound_names(stmt.body, include_for_targets=include_for_targets)
                bound |= self._body_bound_names(
                    stmt.orelse, include_for_targets=include_for_targets
                )
            elif isinstance(stmt, ast.For):
                if include_for_targets and isinstance(stmt.target, ast.Name):
                    bound.add(stmt.target.id)
                bound |= self._body_bound_names(stmt.body, include_for_targets=include_for_targets)
        return bound

    def lower_for(self, stmt: ast.For) -> None:
        """Lower a constant-trip ``for i in range(...)`` (plan D6).

        Loop-carried variables (bound before the loop and reassigned in the
        body) get a fresh loop name with an explicit phi binding
        ``s_loop = s_pre`` just before the loop; body assignments write the
        loop name directly, so the real generated loop feeds values across
        iterations. Constructs that rebind a carried variable (branch merges,
        inner loops) are synchronized back after each body statement.
        """
        iter_call = stmt.iter
        assert isinstance(iter_call, ast.Call)  # validator guarantees range(...)
        start, stop, step = self._loop_bounds(iter_call)

        assert isinstance(stmt.target, ast.Name)  # validator guarantees
        loop_var = stmt.target.id
        loop_name = self.ssa.bind(loop_var)

        pre_definite = dict(self.definite)
        body_assigned = self._body_bound_names(stmt.body, include_for_targets=False)
        body_bound = self._body_bound_names(stmt.body, include_for_targets=True)
        # iterate pre_definite (insertion order) for deterministic phi order;
        # carried includes nested for-targets (they bind in the enclosing
        # scope, so rebinding them is a cross-iteration dependency)
        carried = {v: pre_definite[v] for v in pre_definite if v in body_bound}
        if loop_var in body_assigned:
            raise self.error(stmt, f"cannot assign the loop variable {loop_var!r} inside its loop")

        phi_kinds: dict[str, str | None] = {}
        # a pre-bound loop variable keeps its value on zero-trip loops:
        # emit a phi (the for-statement overwrites it on real iterations)
        if loop_var in pre_definite:
            self.bindings.append(Binding(loop_name, Ref(pre_definite[loop_var])))
            self.name_kinds.setdefault(loop_name, self.name_kinds.get(pre_definite[loop_var]))

        # phis for loop-carried variables, emitted just before the loop
        for var, pre_name in carried.items():
            loop_carried_name = self.ssa.bind(var)
            carried[var] = loop_carried_name
            self.bindings.append(Binding(loop_carried_name, Ref(pre_name)))
            self.definite[var] = loop_carried_name
            phi_kinds[loop_carried_name] = self.name_kinds.get(pre_name)
            self.name_kinds[loop_carried_name] = self.name_kinds.get(pre_name)
            # the phi feeds the pre-loop value on zero-trip loops: a raw
            # scalar stays a raw scalar, so track that for call sites
            if pre_name in self._scalar_names or pre_name in self.name_literals:
                self._scalar_names.add(loop_carried_name)
            if pre_name in self._maybe_bool_names:
                self._maybe_bool_names.add(loop_carried_name)

        phi_end = len(self.bindings)

        def union_labels(
            frame: dict[str, set[str | None]],
        ) -> tuple[dict[str, str | None], list[str]]:
            """Per carried name: the final kind (union of the phi kind and
            all body-assignment kinds) and the names whose kinds MIX across
            iterations."""
            labels: dict[str, str | None] = {}
            mixed: list[str] = []
            for loop_carried_name in carried.values():
                kinds = {phi_kinds.get(loop_carried_name)} | frame.get(loop_carried_name, set())
                unique = {k for k in kinds if k is not None}
                if (None in kinds and unique) or len(unique) > 1:
                    # mixed (or unknown-mixed) kinds across iterations:
                    # label 'bool' so exact-arithmetic intify kicks in at
                    # uses inside the body (the intify casts to the sibling
                    # operand's runtime dtype, so no truncation is possible)
                    labels[loop_carried_name] = "bool"
                    mixed.append(loop_carried_name)
                else:
                    labels[loop_carried_name] = unique.pop() if unique else None
            return labels, mixed

        # The body is lowered repeatedly until per-name kinds reach a fixed
        # point. A carried variable's kind can mix across iterations (int
        # entry, bool body assignment, ...), and the mixing can propagate
        # through assignment chains (a = b; b = c; c = x > 0), so one pass
        # cannot see every name that needs the intify conversion. Labels
        # only ever WIDEN (int/float -> bool), so at most len(carried)
        # widenings can happen; bound the passes at len(carried) + 2.
        labels: dict[str, str | None] = {}
        prev_labels: dict[str, str | None] | None = None
        max_passes = len(carried) + 2
        for attempt in range(max_passes):
            pre_body = (
                dict(self.definite),
                set(self.maybe),
                list(self.deferred),
                dict(self.name_kinds),
                dict(self.name_literals),
            )
            self._carried_assign_kinds.append({})
            self.loop_depth += 1
            self.loop_vars.add(loop_var)
            self.active_carried.append((dict(carried), self.branch_depth))
            self.definite[loop_var] = loop_name
            self.name_kinds.setdefault(loop_name, "int")  # loop vars are Python ints
            self._scalar_names.add(loop_name)  # raw Python ints per iteration
            self.maybe.discard(loop_var)
            try:
                result = self.lower_stmts(stmt.body, loop_body=True)
            finally:
                self.loop_depth -= 1
                self.loop_vars.discard(loop_var)
                self.active_carried.pop()
                frame = self._carried_assign_kinds.pop()
            if result is not None:  # pragma: no cover - returns rejected in bodies
                raise self.error(stmt, "returns inside loop bodies are not supported")
            labels, mixed = union_labels(frame)
            if labels == prev_labels or not mixed or attempt == max_passes - 1:
                break
            # restore the pre-body state and re-lower with the widened
            # labels pre-applied
            prev_labels = labels
            (definite, maybe, deferred, name_kinds, name_literals) = pre_body
            del self.bindings[phi_end:]
            self.definite, self.maybe, self.deferred = definite, maybe, deferred
            self.name_kinds, self.name_literals = name_kinds, name_literals
            self.name_kinds.update(labels)
        for loop_carried_name, kind in labels.items():
            self.name_kinds[loop_carried_name] = kind
        # literal facts are invalid across the loop boundary: the phi feeds
        # pre-loop values on zero-trip loops, and body assignments feed
        # later iterations — neither is the recorded literal
        for loop_carried_name in carried.values():
            self.name_literals.pop(loop_carried_name, None)
        body_stmts = tuple(self.bindings[phi_end:])
        del self.bindings[phi_end:]
        self.bindings.append(Loop(loop_name, start, stop, step, body_stmts))

        # loop-local variables (assigned in body, not bound before) keep their
        # body names: zero-trip loops raise NameError exactly like Python.
        # The loop variable stays bound after the loop (last iteration value).

    def _sync_loop_carried(self) -> None:
        """Copy rebound values back to their active loop-carried names."""
        for carried, _entry_depth in reversed(self.active_carried):
            for var, loop_name in carried.items():
                current = self.definite.get(var)
                if current is not None and current != loop_name:
                    self.bindings.append(Binding(loop_name, Ref(current)))
                    self.definite[var] = loop_name
                    self.name_kinds[loop_name] = self.name_kinds.get(current)
                    self.name_literals.pop(loop_name, None)

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
        cond: Node,
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
                merged_expr: Node = Where(cond, Ref(nt), Ref(ne))
                self.bindings.append(Binding(name, merged_expr))
                self.definite[var] = name
                kt, ke = self.name_kinds.get(nt), self.name_kinds.get(ne)
                self.name_kinds[name] = kt if kt == ke else None
                if self._maybe_bool_result(merged_expr):
                    self._maybe_bool_names.add(name)
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

    def fold_returns(self, value: Node) -> Node:
        """Fold pending early returns around ``value``.

        Entries pushed at the current branch depth or deeper belong to this
        block: they are applied here (list order is divergence order, so the
        fold nests earlier returns outermost). Entries pushed at shallower
        depths belong to enclosing blocks and are retained for their folds;
        each branch copy carries them, and mixed ifs insert their own
        divergence before enclosed entries.
        """
        result = value
        retained: list[tuple[Node, Node, int]] = []
        for cond, early, depth in reversed(self.deferred):
            if depth < self.branch_depth:
                retained.append((cond, early, depth))
            else:
                result = Where(cond, early, result)
        retained.reverse()
        self.deferred = retained
        return result

    # --------------------------------------------------------- expressions

    def lower_expr(self, node: ast.expr) -> Node:
        if isinstance(node, ast.Constant):
            return self._literal(node.value)
        if isinstance(node, ast.Name):
            return self.load_name(node)
        if isinstance(node, ast.BinOp):
            return self._lower_binop(
                node.op, self.lower_expr(node.left), self.lower_expr(node.right)
            )
        if isinstance(node, ast.UnaryOp):
            operand = self.lower_expr(node.operand)
            if isinstance(node.op, ast.USub) and (
                self._numeric_kind(operand) == "bool" or self._maybe_bool_result(operand)
            ):
                # -True is -1 in Python; numpy cannot negate bool arrays.
                # The whole negation goes through the runtime helper (it
                # computes the negation — no wrapping UnaryOp on top)
                if isinstance(operand, Literal):
                    return Literal(-int(operand.value), "int")
                return FuncCall(
                    self.arith_name,
                    (Ref(self.ns_var), Literal(_ARITH_OPS.index("neg"), "int"), operand),
                )
            return UnaryOp(_UNARY[type(node.op)], operand)
        if isinstance(node, ast.Compare):
            return self._lower_compare(node)
        if isinstance(node, ast.BoolOp):
            return self._lower_boolop(node)
        if isinstance(node, ast.IfExp):
            return Where(
                self._coerce_bool(self.lower_expr(node.test)),
                self.lower_expr(node.body),
                self.lower_expr(node.orelse),
            )
        if isinstance(node, ast.Call):
            return self._lower_call(node)
        if isinstance(node, ast.Attribute):
            return self._lower_math_const(node)
        raise self.error(node, f"{type(node).__name__} expressions are not supported")

    def _lower_compare(self, node: ast.Compare) -> Node:
        parts: list[Node] = []
        operands = [node.left, *node.comparators]
        for left, op, right in zip(operands, node.ops, operands[1:], strict=False):
            parts.append(Compare(_CMPOPS[type(op)], self.lower_expr(left), self.lower_expr(right)))
        if len(parts) == 1:
            return parts[0]
        return Logical("and", tuple(parts))

    def _lower_boolop(self, node: ast.BoolOp) -> Node:
        op = "and" if isinstance(node.op, ast.And) else "or"
        values = [self.lower_expr(v) for v in node.values]
        if all(is_bool(v) for v in values):
            return Logical(op, tuple(values))
        # exact value-select lowering (plan D2)
        partial = values[0]
        last = len(values) - 1
        for i, value in enumerate(values[1:], 1):
            if op == "and":
                merged: Node = Where(self._coerce_bool(partial), value, partial)
            else:
                merged = Where(self._coerce_bool(partial), partial, value)
            if i < last:
                # bind intermediates to temps; only the final value stays inline
                temp = self.ssa.fresh_temp("v")
                self.bindings.append(Binding(temp, merged))
                partial = Ref(temp)
            else:
                partial = merged
        return partial

    def _lower_math_const(self, node: ast.Attribute) -> Node:
        return self._literal(MATH_CONSTS[node.attr])

    def _lower_call(self, node: ast.Call) -> Node:
        args = [self.lower_expr(a) for a in node.args]
        func = node.func
        if isinstance(func, ast.Attribute):
            return self._call_math(func, func.attr, args, node)
        assert isinstance(func, ast.Name)
        name = func.id
        info = self.info
        if name in self.definite or name in self.maybe or name in self.assigned_names:
            raise self.error(node, f"calling variable {name!r} is not supported")
        if name in info.math_funcs:
            return self._call_math(func, info.math_funcs[name], args, node)
        if name in info.user_funcs:
            return self._call_helper(node, name, args)
        if name in info.math_modules:
            raise self.error(node, "the math module cannot be called")
        if name == "round" and len(args) != 1:
            raise self.error(
                node,
                "round(x, n) is not supported: the decimals kwarg is not in the Array API standard",
            )
        if name in BUILTIN_UNARY:
            self._check_arity(node, name, args, 1, 1)
            sanitized = tuple(self._sanitize_xp_arg(a, force_float=False) for a in args)
            return Call(BUILTIN_UNARY[name], sanitized)
        if name in BUILTIN_FOLDS:
            if len(args) < 2:
                raise self.error(node, f"{name}() needs at least 2 arguments in vectorized code")
            if all(self._literal_value(a) is not None for a in args):
                # all-literal call: substitute the actual literals so
                # const-folding fires (Ref nodes would block it and reach
                # codegen as raw Python scalars)
                clean: list[Node] = [
                    lit for a in args if (lit := self._literal_value(a)) is not None
                ]
            else:
                # one runtime helper call with exact selection semantics:
                # identical-dtype fast paths, never-winning-bound clamps,
                # proven integer result ranges (max(uint64, signed) fits
                # uint64; min(uint64, signed) fits int64), Python float
                # promotion. Literal args pass as raw values; array args
                # pass through the soft sanitizer
                is_min = BUILTIN_FOLDS[name] == "minimum"
                helper_args: list[Node] = [Ref(self.ns_var), Literal(is_min, "bool")]
                for a in args:
                    lit = self._literal_value(a)
                    if lit is None:
                        helper_args.append(self._sanitize_xp_arg(a, force_float=False))
                    else:
                        helper_args.append(lit)
                return FuncCall(self.minmax_name, tuple(helper_args))
            folded: Node = Call(BUILTIN_FOLDS[name], (clean[0], clean[1]))
            for arg in clean[2:]:
                folded = Call(BUILTIN_FOLDS[name], (folded, arg))
            return folded
        if name in BUILTIN_CASTS:
            self._check_arity(node, name, args, 1, 1)
            return self._cast_call(args[0], BUILTIN_CASTS[name])
        raise self.error(node, f"unsupported function {name!r}")

    def _cast_call(self, arg: Node, dtype: str) -> Node:
        """int()/float()/bool()/math.trunc-style casts: fold literal args
        (exact), wrap raw loop-variable ints in asarray, pass arrays
        through (params are preamble-normalized)."""
        lit = self._literal_value(arg)
        if lit is not None:
            try:
                if dtype == "bool":
                    return Literal(bool(lit.value), "bool")
                if dtype == "int64":
                    return Literal(int(lit.value), "int")
                return Literal(float(lit.value), "float")
            except (ValueError, OverflowError):
                pass  # inf/nan literals: leave the cast to runtime astype
        return Call("astype", (self._sanitize_xp_arg(arg, force_float=False), DType(dtype)))

    def _call_math(self, func: ast.AST, attr: str, args: list[Node], node: ast.AST) -> Node:
        if attr in MATH_SPECIAL:
            self._check_arity(node, attr, args, 1, 1)
            return self._cast_call(args[0], MATH_SPECIAL[attr])
        if attr in MATH_FUNCS:
            xp_name, lo, hi = MATH_FUNCS[attr]
            self._check_arity(node, attr, args, lo, hi)
            # Array API elementwise math requires floating-point array
            # inputs; sanitize literal/int/bool/possibly-scalar args
            sanitized = tuple(self._sanitize_xp_arg(a, force_float=True) for a in args)
            return Call(xp_name, sanitized)
        raise self.error(func, f"math.{attr} cannot be called")

    @staticmethod
    def _is_negative_int(node: Node) -> bool:
        """True for literal negative ints (including UnaryOp(neg, lit))."""
        if isinstance(node, Literal) and node.kind == "int":
            return node.value < 0
        return (
            isinstance(node, UnaryOp)
            and node.op == "neg"
            and isinstance(node.operand, Literal)
            and node.operand.kind == "int"
        )

    def _numeric_kind(self, node: Node) -> str | None:
        """Best-effort numeric kind: literals, casts, bools, arithmetic, and
        Refs whose binding kind was recorded are provable."""
        if isinstance(node, Literal):
            return node.kind
        if isinstance(node, Ref):
            return self.name_kinds.get(node.name)
        if isinstance(node, Logical):
            return "bool"
        if isinstance(node, Where):
            kt, ke = self._numeric_kind(node.then), self._numeric_kind(node.other)
            return kt if kt == ke else None
        if is_bool(node):
            return "bool"
        if isinstance(node, BinOp):
            # scalar promotion rules: int/bool + float -> float; arithmetic
            # over ints/bools stays int; true division is ALWAYS float in
            # Python (int / int -> float), whatever the operand kinds
            lk, rk = self._numeric_kind(node.left), self._numeric_kind(node.right)
            if node.op == "div":
                return "float"
            if node.op in ("and", "or", "xor"):
                # bitwise: Python bool & bool is bool (saturating array bool
                # arithmetic would otherwise lose exactness downstream)
                if lk == "bool" and rk == "bool":
                    return "bool"
                if lk in ("int", "bool") and rk in ("int", "bool"):
                    return "int"
                return None
            if "float" in (lk, rk):
                return "float"
            if lk in ("int", "bool") and rk in ("int", "bool"):
                return "int"
            return None
        if isinstance(node, UnaryOp) and node.op in ("neg", "pos"):
            return self._numeric_kind(node.operand)
        if isinstance(node, Call) and node.fn == "asarray" and len(node.args) == 1:
            # asarray preserves the wrapped value's kind
            return self._numeric_kind(node.args[0])
        if isinstance(node, FuncCall) and node.fn == self.minmax_name:
            # min/max result kind: bool when every argument is bool, float
            # when any argument is float, int for int-ish mixes (the helper
            # normalizes bool arrays to int8, so arithmetic downstream must
            # intify)
            kinds = [self._numeric_kind(a) for a in node.args[2:]]
            if any(k is None for k in kinds):
                return None
            if "float" in kinds:
                return "float"
            if all(k == "bool" for k in kinds):
                return "bool"
            return "int"
        if (
            isinstance(node, Call)
            and node.fn == "astype"
            and len(node.args) == 2
            and isinstance(node.args[1], DType)
        ):
            name = node.args[1].name
            return "bool" if name == "bool" else "int" if name == "int64" else "float"
        return None

    def _lower_binop(self, op: ast.operator, left: Node, right: Node) -> Node:
        """Build a BinOp with the exactness fixes shared by `a op b` and `a op= b`."""
        # Python bool arithmetic is integer (True + True == 2), but array
        # backends either saturate (bool + bool == True) or reject bools in
        # arithmetic outright (strict backends). Cast bool-typed operands to
        # float64: exact for bools (0/1) and ints up to 2**53, and a no-op
        # for floats — so loop-carried variables whose kind changes across
        # iterations stay correct too.
        if isinstance(op, ast.Add | ast.Sub | ast.Mult | ast.Div | ast.FloorDiv | ast.Mod) and (
            "bool" in (self._numeric_kind(left), self._numeric_kind(right))
            or self._maybe_bool_result(left)
            or self._maybe_bool_result(right)
        ):
            # the whole operation goes through the runtime _vec_arith
            # helper: bools need a numeric representation, strict
            # backends require single-dtype or float operands, and the
            # uint64 integer paths are exact per lane (modular for
            # add/sub/mul; int64 magnitudes for negative divisors)
            ir_op = _BINOPS[type(op)]
            return FuncCall(
                self.arith_name,
                (Ref(self.ns_var), Literal(_ARITH_OPS.index(ir_op), "int"), left, right),
            )
        if isinstance(op, ast.Pow) and self._is_negative_int(right):
            # Python promotes int ** negative-int to float; NumPy raises.
            # Cast provably-int bases so the common literal case matches.
            left = self._as_float_arg(left)
        return BinOp(_BINOPS[type(op)], left, right)

    def _literal_value(self, arg: Node) -> Literal | None:
        """The literal behind an arg: directly, as a negated literal
        (``-1`` lowers to UnaryOp(neg, 1)), or via a literal binding."""
        if isinstance(arg, Literal):
            return arg
        if (
            isinstance(arg, UnaryOp)
            and arg.op == "neg"
            and isinstance(arg.operand, Literal)
            and arg.operand.kind in ("int", "float")
        ):
            return Literal(-arg.operand.value, arg.operand.kind)
        if isinstance(arg, Ref):
            return self.name_literals.get(arg.name)
        return None

    def _maybe_bool_result(self, node: Node) -> bool:
        """True for min/max results whose static kind is unknown: their
        runtime dtype may be boolean (parameter operands), so arithmetic
        must use the runtime-polymorphic intify — the _vec_arith_dtype
        cast is a no-op for numeric dtypes and int64 for booleans."""

        if isinstance(node, FuncCall) and node.fn == self.minmax_name:
            return self._numeric_kind(node) is None
        if isinstance(node, Ref):
            return node.name in self._maybe_bool_names
        if isinstance(node, Where):
            # branch merge of maybe-bool values is maybe-bool
            return self._maybe_bool_result(node.then) or self._maybe_bool_result(node.other)
        if isinstance(node, BinOp) and node.op in ("and", "or", "xor"):
            # bitwise expressions preserve boolean-ness (arithmetic
            # results are already intified and numeric)
            return self._maybe_bool_result(node.left) or self._maybe_bool_result(node.right)
        return False

    def _possibly_scalar(self, node: Node) -> bool:
        """True when the node's runtime value may be a raw Python scalar:
        it is a literal, a possibly-scalar name (parameter default, loop
        variable, literal binding), or an expression computed from one.
        Operators promote such scalars fine; xp.* function arguments do
        not (strict backends reject plain scalars)."""

        def is_scalar_name(name: str) -> bool:
            return name in self._scalar_names or name in self.name_literals

        def walk(n: Node) -> bool:
            if isinstance(n, Literal):
                return True
            if isinstance(n, Ref):
                return is_scalar_name(n.name)
            if isinstance(n, DType):
                return False
            return any(walk(c) for c in _children(n))

        return walk(node)

    def _sanitize_xp_arg(self, arg: Node, *, force_float: bool) -> Node:
        """Make an argument safe for xp.* calls.

        Operators promote raw Python scalars fine, but xp.* function
        arguments must be arrays (strict backends reject plain scalars) —
        and elementwise math requires floating-point arrays on every
        backend.

        ``force_float`` (math functions): scalar semantics convert to
        double, so literals become float literals (const-folded away) and
        provably int/bool or possibly-scalar args are cast to float64.

        Otherwise (polymorphic builtins such as abs/min/max and dtype
        casts): keep values exact — literals and possibly-scalar args are
        wrapped in asarray, with no dtype conversion.
        """
        lit = self._literal_value(arg)
        if lit is not None:
            if force_float:
                return Literal(float(lit.value), "float")
            return Call("asarray", (lit,))
        kind = self._numeric_kind(arg)
        if kind in ("int", "bool") or self._possibly_scalar(arg):
            if force_float:
                return Call("astype", (Call("asarray", (arg,)), DType("float64")))
            return Call("asarray", (arg,))
        return arg

    def _as_float_arg(self, arg: Node) -> Node:
        kind = self._numeric_kind(arg)
        if kind in ("int", "bool"):
            if isinstance(arg, Literal):
                return Literal(float(arg.value), "float")
            return Call("astype", (Call("asarray", (arg,)), DType("float64")))
        return arg

    def _call_helper(self, node: ast.Call, name: str, args: list[Node]) -> Node:
        """Vectorize and call another scalar function (plan D7)."""
        if self.helper_vectorizer is None:
            raise self.error(
                node,
                f"call to {name!r} requires the vectorize pipeline "
                "(helper functions are only supported through vectorize())",
            )
        callee = self.info.user_funcs[name]
        if callee in self._helper_names:
            return FuncCall(self._helper_names[callee], tuple(args))
        base = generated_name(callee.__name__)
        while not self.ssa.is_free(base):
            base += "_"
        self.ssa.reserve(base)
        vec = self.helper_vectorizer(callee)
        self.helpers.append((base, vec))
        self._helper_names[callee] = base
        return FuncCall(base, tuple(args))

    def _check_arity(self, node: ast.AST, name: str, args: list[Node], lo: int, hi: int) -> None:
        if not lo <= len(args) <= hi:
            expected = str(lo) if lo == hi else f"{lo}-{hi}"
            raise self.error(node, f"{name}() takes {expected} argument(s), got {len(args)}")


def lower_function(
    info: FunctionInfo,
    helper_vectorizer: HelperVectorizer | None = None,
) -> LoweredFunction:
    """Lower a validated FunctionInfo to IR (plan §5/§6/§7)."""
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
    for attr, base, fn in (
        ("minmax_name", "_vec_minmax", _vec_minmax),
        ("arith_name", "_vec_arith", _vec_arith),
    ):
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
        lowerer._scalar_names.add(emitted)
    hidden: list[tuple[str, Any]] = []
    for var, array in info.closure_arrays.items():
        emitted = lowerer.ssa.bind(var)
        lowerer.definite[var] = emitted
        hidden.append((emitted, array))

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
        helpers=lowerer.helpers,
        namespace_var=ns,
        emitted_names=frozenset(lowerer.ssa.emitted),
    )
