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
        self.name_kinds[name] = self._numeric_kind(value)
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

    def _body_assigned_names(self, stmts: list[ast.stmt]) -> set[str]:
        """Names assigned via ``=``/augmented assignment in a statement list.

        ``for`` targets are excluded: nested loops bind their own variable.
        """
        assigned: set[str] = set()
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        assigned.add(t.id)
            elif isinstance(stmt, ast.AugAssign):
                if isinstance(stmt.target, ast.Name):
                    assigned.add(stmt.target.id)
            elif isinstance(stmt, ast.If):
                assigned |= self._body_assigned_names(stmt.body)
                assigned |= self._body_assigned_names(stmt.orelse)
            elif isinstance(stmt, ast.For):
                assigned |= self._body_assigned_names(stmt.body)
        return assigned

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
        body_assigned = self._body_assigned_names(stmt.body)
        # iterate pre_definite (insertion order) for deterministic phi order
        carried = {v: pre_definite[v] for v in pre_definite if v in body_assigned}
        if loop_var in body_assigned:
            raise self.error(stmt, f"cannot assign the loop variable {loop_var!r} inside its loop")

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
            self.name_kinds[loop_carried_name] = self.name_kinds.get(pre_name)

        phi_end = len(self.bindings)
        self.loop_depth += 1
        self.loop_vars.add(loop_var)
        self.active_carried.append((dict(carried), self.branch_depth))
        self.definite[loop_var] = loop_name
        self.name_kinds.setdefault(loop_name, None)
        self.maybe.discard(loop_var)
        try:
            result = self.lower_stmts(stmt.body, loop_body=True)
        finally:
            self.loop_depth -= 1
            self.loop_vars.discard(loop_var)
            self.active_carried.pop()
        if result is not None:  # pragma: no cover - returns rejected in bodies
            raise self.error(stmt, "returns inside loop bodies are not supported")
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
            if isinstance(node.op, ast.USub) and self._numeric_kind(operand) == "bool":
                # -True is -1 in Python; numpy cannot negate bool arrays
                operand = self._as_int_arg(operand)
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
            return Call(BUILTIN_UNARY[name], tuple(args))
        if name in BUILTIN_FOLDS:
            if len(args) < 2:
                raise self.error(node, f"{name}() needs at least 2 arguments in vectorized code")
            folded: Node = Call(BUILTIN_FOLDS[name], (args[0], args[1]))
            for arg in args[2:]:
                folded = Call(BUILTIN_FOLDS[name], (folded, arg))
            return folded
        if name in BUILTIN_CASTS:
            self._check_arity(node, name, args, 1, 1)
            return Call("astype", (args[0], DType(BUILTIN_CASTS[name])))
        raise self.error(node, f"unsupported function {name!r}")

    def _call_math(self, func: ast.AST, attr: str, args: list[Node], node: ast.AST) -> Node:
        if attr in MATH_SPECIAL:
            self._check_arity(node, attr, args, 1, 1)
            return Call("astype", (args[0], DType(MATH_SPECIAL[attr])))
        if attr in MATH_FUNCS:
            xp_name, lo, hi = MATH_FUNCS[attr]
            self._check_arity(node, attr, args, lo, hi)
            # Array API elementwise math requires floating-point inputs;
            # provably int/bool arguments are cast (plan D9 backend neutrality)
            floated = tuple(self._as_float_arg(a) for a in args)
            return Call(xp_name, floated)
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
        """Best-effort numeric kind: literals, casts, bools, and Refs whose
        binding kind was recorded are provable."""
        if isinstance(node, Literal):
            return node.kind
        if isinstance(node, Ref):
            return self.name_kinds.get(node.name)
        if is_bool(node):
            return "bool"
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
        # bool arithmetic saturates (True + True == True): cast provably-bool
        # operands to int64 for exact semantics.
        if (
            isinstance(op, ast.Add | ast.Sub | ast.Mult | ast.Div | ast.FloorDiv | ast.Mod)
            and self._numeric_kind(left) == "bool"
            and self._numeric_kind(right) == "bool"
        ):
            left = self._as_int_arg(left)
            right = self._as_int_arg(right)
        if isinstance(op, ast.Pow) and self._is_negative_int(right):
            # Python promotes int ** negative-int to float; NumPy raises.
            # Cast provably-int bases so the common literal case matches.
            left = self._as_float_arg(left)
        return BinOp(_BINOPS[type(op)], left, right)

    def _as_int_arg(self, arg: Node) -> Node:
        if isinstance(arg, Literal):
            return Literal(int(arg.value), "int")
        return Call("astype", (arg, DType("int64")))

    def _as_float_arg(self, arg: Node) -> Node:
        kind = self._numeric_kind(arg)
        if kind in ("int", "bool"):
            if isinstance(arg, Literal):
                return Literal(float(arg.value), "float")
            return Call("astype", (arg, DType("float64")))
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


def _namespace_var(info: FunctionInfo) -> str:
    """Pick the namespace variable name: ``xp`` unless the user uses it."""
    if "xp" not in info.user_names:
        return "xp"
    i = 1
    while f"xp_{i}" in info.user_names:
        i += 1
    return f"xp_{i}"


def lower_function(
    info: FunctionInfo,
    helper_vectorizer: HelperVectorizer | None = None,
) -> LoweredFunction:
    """Lower a validated FunctionInfo to IR (plan §5/§6/§7)."""
    lowerer = _Lowerer(info, helper_vectorizer=helper_vectorizer)
    # (5) the caller's own generated name is reserved so a same-named helper
    # cannot collide with it in the runtime namespace
    lowerer.ssa.reserve(generated_name(info.name))
    # bind parameter names in order (first bind keeps the name, mangling reserved)
    param_names = [lowerer.ssa.bind(p.name) for p in info.params]
    for param, emitted in zip(info.params, param_names, strict=True):
        lowerer.definite[param.name] = emitted
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
        namespace_var=_namespace_var(info),
        emitted_names=frozenset(lowerer.ssa.emitted),
    )
