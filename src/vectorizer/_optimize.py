"""Optimizer passes: const-fold, CSE, DCE (plan §8)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any

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
)

__all__ = ["optimize"]

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1

#: pow is never folded (backend pow semantics on edge values differ);
#: div/mod/floordiv by literal zero are skipped (runtime inf/NaN, plan D3).
_NO_FOLD_OPS = frozenset({"pow"})
_ZERO_RISK_OPS = frozenset({"div", "floordiv", "mod"})


def _int64_ok(value: int) -> bool:
    return _INT64_MIN <= value <= _INT64_MAX


def _literal_kind(value: object) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    return "float"


def _fold_binop(op: str, left: Literal, right: Literal) -> Literal | None:
    if op in _NO_FOLD_OPS:
        return None
    if op in _ZERO_RISK_OPS and right.value == 0:
        return None
    a: Any = left.value
    b: Any = right.value
    try:
        if op == "add":
            result: object = a + b
        elif op == "sub":
            result = a - b
        elif op == "mul":
            result = a * b
        elif op == "div":
            result = a / b
        elif op == "floordiv":
            result = a // b
        elif op == "mod":
            result = a % b
        elif op == "and":
            result = a & b
        elif op == "or":
            result = a | b
        elif op == "xor":
            result = a ^ b
        elif op == "lshift":
            result = a << b
        elif op == "rshift":
            result = a >> b
        else:
            return None
    except (ArithmeticError, ValueError, TypeError):
        return None
    if isinstance(result, int) and not isinstance(result, bool) and not _int64_ok(result):
        return None
    return Literal(result, _literal_kind(result))  # type: ignore[arg-type]


def _fold_unary(op: str, lit: Literal) -> Literal | None:
    v: Any = lit.value
    if op == "not":
        # exact per D2: logical_not(NaN) is False, Python `not nan` is False
        return Literal(not v, "bool")
    try:
        if op == "neg":
            result: Any = -v
        elif op == "pos":
            result = +v
        elif op == "invert":
            result = ~v
        else:
            return None
    except (ArithmeticError, ValueError, TypeError):
        return None
    if isinstance(result, int) and not isinstance(result, bool) and not _int64_ok(result):
        return None
    return Literal(result, _literal_kind(result))


# ------------------------------------------------------------------ rewriting


def _rewrite(node: Node, fn: Callable[[Node], Node]) -> Node:
    """Rebuild ``node`` with ``fn`` applied to each rewritten child."""
    if isinstance(node, Literal | Ref | DType):
        return node
    if isinstance(node, BinOp):
        return BinOp(node.op, fn(node.left), fn(node.right))
    if isinstance(node, UnaryOp):
        return UnaryOp(node.op, fn(node.operand))
    if isinstance(node, Compare):
        return Compare(node.op, fn(node.left), fn(node.right))
    if isinstance(node, Logical):
        return Logical(node.op, tuple(fn(p) for p in node.parts))
    if isinstance(node, Where):
        return Where(fn(node.cond), fn(node.then), fn(node.other))
    if isinstance(node, Call):
        return Call(node.fn, tuple(fn(a) for a in node.args))
    raise TypeError(f"unexpected IR node {type(node).__name__}")


def _children(node: Node) -> tuple[Node, ...]:
    if isinstance(node, Literal | Ref | DType):
        return ()
    if isinstance(node, BinOp):
        return (node.left, node.right)
    if isinstance(node, UnaryOp):
        return (node.operand,)
    if isinstance(node, Compare):
        return (node.left, node.right)
    if isinstance(node, Logical):
        return node.parts
    if isinstance(node, Where):
        return (node.cond, node.then, node.other)
    if isinstance(node, Call):
        return node.args
    raise TypeError(f"unexpected IR node {type(node).__name__}")


def _const_fold_expr(node: Node) -> Node:
    if isinstance(node, BinOp):
        node = BinOp(node.op, _const_fold_expr(node.left), _const_fold_expr(node.right))
        if isinstance(node.left, Literal) and isinstance(node.right, Literal):
            folded = _fold_binop(node.op, node.left, node.right)
            if folded is not None:
                return folded
        return node
    if isinstance(node, UnaryOp):
        node = UnaryOp(node.op, _const_fold_expr(node.operand))
        if isinstance(node.operand, Literal):
            folded = _fold_unary(node.op, node.operand)
            if folded is not None:
                return folded
        return node
    return _rewrite(node, _const_fold_expr)


def const_fold(program: Program) -> Program:
    bindings = tuple(Binding(b.name, _const_fold_expr(b.expr)) for b in program.bindings)
    return Program(program.params, bindings, _const_fold_expr(program.result))


# ----------------------------------------------------------------------- DCE


def _mark_live(node: Node, live: set[str]) -> None:
    if isinstance(node, Ref):
        live.add(node.name)
    for child in _children(node):
        _mark_live(child, live)


def dce(program: Program) -> Program:
    live: set[str] = set()
    _mark_live(program.result, live)
    kept: list[Binding] = []
    # single reverse pass suffices: SSA uses only earlier bindings
    for binding in reversed(program.bindings):
        if binding.name in live:
            kept.append(binding)
            _mark_live(binding.expr, live)
    kept.reverse()
    return Program(program.params, tuple(kept), program.result)


# ----------------------------------------------------------------------- CSE

_CSE_ELIGIBLE = (Call, Where)


def _count_eligible(node: Node, counter: Counter[Node]) -> None:
    if isinstance(node, _CSE_ELIGIBLE):
        counter[node] += 1
    for child in _children(node):
        _count_eligible(child, counter)


def cse(program: Program, ssa: SSAEnv) -> Program:
    """Hoist duplicated Call/Where subtrees into temps ``t_1, t_2, ...``.

    Each pass counts eligible nodes in the *current* program and hoists those
    seen at least twice, inserting each temp just before the first binding
    that uses it (all bindings are at function scope, so hoisting is safe).
    Rebuilt parents (whose children became temps) only become countable in
    the next pass, hence the fixpoint loop.
    """
    current = program
    for _ in range(8):
        counter: Counter[Node] = Counter()
        for binding in current.bindings:
            _count_eligible(binding.expr, counter)
        _count_eligible(current.result, counter)
        if not any(count >= 2 for count in counter.values()):
            return current

        memo: dict[Node, str] = {}
        pending: list[Binding] = []

        def rewrite(
            node: Node,
            *,
            counter: Counter[Node] = counter,
            memo: dict[Node, str] = memo,
            pending: list[Binding] = pending,
        ) -> Node:
            node = _rewrite(node, rewrite)
            if isinstance(node, _CSE_ELIGIBLE) and counter[node] >= 2:
                if node not in memo:
                    temp = ssa.fresh_temp("t")
                    pending.append(Binding(temp, node))
                    memo[node] = temp
                return Ref(memo[node])
            return node

        rewritten: list[Binding] = []
        for binding in current.bindings:
            start = len(pending)
            expr = rewrite(binding.expr)
            rewritten.extend(pending[start:])
            rewritten.append(Binding(binding.name, expr))
        result_start = len(pending)
        result = rewrite(current.result)
        rewritten.extend(pending[result_start:])  # temps first used by the result go last
        current = Program(current.params, tuple(rewritten), result)
    return current


def optimize(program: Program, user_names: set[str] | None = None) -> Program:
    """Run const-fold, CSE, DCE (plan §8)."""
    folded = const_fold(program)
    ssa = SSAEnv(user_names or set())
    deduped = cse(folded, ssa)
    return dce(deduped)
