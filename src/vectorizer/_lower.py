"""AST -> IR lowering (plan §5 translation rules, §6 IR, §7 semantics).

Handles straight-line code (M1) and if/elif/else with early returns (M2).
Loops and helper calls are gated pending M3.
"""

from __future__ import annotations

import ast
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
    Literal,
    Logical,
    Node,
    Program,
    Ref,
    SSAEnv,
    UnaryOp,
    Where,
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


class _Lowerer:
    def __init__(self, info: FunctionInfo) -> None:
        self.info = info
        self.ssa = SSAEnv(info.user_names)
        self.bindings: list[Binding] = []
        #: definite locals: var -> current emitted name
        self.definite: dict[str, str] = {}
        #: vars that may be unbound on the current path (assigned in one branch only)
        self.maybe: set[str] = set()
        #: pending early returns: (condition, value, push_depth) in divergence
        #: order; folds nest earlier entries outermost
        self.deferred: list[tuple[Node, Node, int]] = []
        #: nesting depth inside branches (branch-local bindings always suffix)
        self.branch_depth = 0
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

    def lower_stmts(self, stmts: list[ast.stmt]) -> Node | None:
        """Lower a statement list; returns the merged result if a return was hit."""
        for stmt in stmts:
            if isinstance(stmt, ast.Assign | ast.AugAssign):
                self.lower_assign(stmt)
            elif isinstance(stmt, ast.Return):
                assert stmt.value is not None  # validator rejects bare returns
                return self.fold_returns(self.lower_expr(stmt.value))
            elif isinstance(stmt, ast.If):
                merged = self.lower_if(stmt)
                if merged is not None:
                    return merged
            elif isinstance(stmt, ast.For):
                raise self.error(stmt, "for loops are not supported yet (milestone M3)")
            elif isinstance(stmt, ast.Expr | ast.Pass):
                continue  # docstring / pass
            else:
                raise self.error(stmt, f"{type(stmt).__name__} statements are not supported")
        return None

    def lower_assign(self, stmt: ast.Assign | ast.AugAssign) -> None:
        if isinstance(stmt, ast.Assign):
            target = stmt.targets[0]
            assert isinstance(target, ast.Name)  # validator guarantees
            value = self.lower_expr(stmt.value)
        else:
            target = stmt.target
            assert isinstance(target, ast.Name)
            op = _BINOPS[type(stmt.op)]
            value = BinOp(op, self.load_name(target), self.lower_expr(stmt.value))
        name = self.ssa.bind(target.id, force_suffix=self.branch_depth > 0)
        self.bindings.append(Binding(name, value))
        self.definite[target.id] = name
        self.maybe.discard(target.id)

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
        for var in then_def.keys() | else_def.keys():
            nt, ne = then_def.get(var), else_def.get(var)
            if nt == ne:
                continue
            if nt is not None and ne is not None:
                name = self.ssa.bind(var)
                self.bindings.append(Binding(name, Where(cond, Ref(nt), Ref(ne))))
                self.definite[var] = name
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
            return BinOp(
                _BINOPS[type(node.op)], self.lower_expr(node.left), self.lower_expr(node.right)
            )
        if isinstance(node, ast.UnaryOp):
            return UnaryOp(_UNARY[type(node.op)], self.lower_expr(node.operand))
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
            raise self.error(
                node, f"calls to other functions ({name!r}) are not supported yet (milestone M3)"
            )
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
            return Call(xp_name, tuple(args))
        raise self.error(func, f"math.{attr} cannot be called")

    def _check_arity(self, node: ast.AST, name: str, args: list[Node], lo: int, hi: int) -> None:
        if not lo <= len(args) <= hi:
            expected = str(lo) if lo == hi else f"{lo}-{hi}"
            raise self.error(node, f"{name}() takes {expected} argument(s), got {len(args)}")


def lower_function(info: FunctionInfo) -> LoweredFunction:
    """Lower a validated FunctionInfo to IR (plan §5/§6/§7)."""
    lowerer = _Lowerer(info)
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
    )
