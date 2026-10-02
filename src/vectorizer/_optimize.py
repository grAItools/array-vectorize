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
    if isinstance(node, FuncCall):
        return FuncCall(node.fn, tuple(fn(a) for a in node.args))
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
    if isinstance(node, FuncCall):
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


def _fold_stmt(stmt: Stmt) -> Stmt:
    if isinstance(stmt, Binding):
        return Binding(stmt.name, _const_fold_expr(stmt.expr))
    return Loop(
        stmt.var,
        _const_fold_expr(stmt.start),
        _const_fold_expr(stmt.stop),
        _const_fold_expr(stmt.step),
        tuple(_fold_stmt(s) for s in stmt.body),
    )


def const_fold(program: Program) -> Program:
    bindings = tuple(_fold_stmt(s) for s in program.bindings)
    return Program(program.params, bindings, _const_fold_expr(program.result))


# ----------------------------------------------------------------------- DCE


def _mark_live(node: Node, live: set[str]) -> None:
    if isinstance(node, Ref):
        live.add(node.name)
    for child in _children(node):
        _mark_live(child, live)


def _mark_live_stmt(stmt: Stmt, live: set[str]) -> Stmt | Literal[False]:
    """Mark liveness from a statement; returns the kept statement or False."""
    if isinstance(stmt, Binding):
        if stmt.name in live:
            _mark_live(stmt.expr, live)
            return stmt
        return False
    # Loop: process body in reverse; keep if any body statement is kept or
    # the loop variable is live afterwards.
    kept_any = False
    kept: list[Stmt] = []
    for inner in reversed(stmt.body):
        if _mark_live_stmt(inner, live):
            kept.append(inner)
            kept_any = True
    kept.reverse()
    _mark_live(stmt.start, live)
    _mark_live(stmt.stop, live)
    _mark_live(stmt.step, live)
    if kept_any or stmt.var in live:
        return Loop(stmt.var, stmt.start, stmt.stop, stmt.step, tuple(kept))
    return False


def dce(program: Program) -> Program:
    live: set[str] = set()
    _mark_live(program.result, live)
    kept: list[Stmt] = []
    # single reverse pass suffices: SSA uses only earlier bindings
    for stmt in reversed(program.bindings):
        kept_stmt = _mark_live_stmt(stmt, live)
        if kept_stmt is not False:
            kept.append(kept_stmt)
    kept.reverse()
    return Program(program.params, tuple(kept), program.result)


# ----------------------------------------------------------------------- CSE

_CSE_ELIGIBLE = (Call, Where)


def _count_eligible(node: Node, counter: Counter[Node]) -> None:
    if isinstance(node, _CSE_ELIGIBLE):
        counter[node] += 1
    for child in _children(node):
        _count_eligible(child, counter)


def _has_loop(stmts: tuple[Stmt, ...] | list[Stmt]) -> bool:
    for stmt in stmts:
        if isinstance(stmt, Loop):
            if _has_loop(stmt.body):
                return True
            return True
    return False


def cse(program: Program, ssa: SSAEnv) -> Program:
    """Hoist duplicated Call/Where subtrees into temps ``t_1, t_2, ...``.

    Each pass counts eligible nodes in the *current* program and hoists those
    seen at least twice, inserting each temp just before the first binding
    that uses it (all bindings are at function scope, so hoisting is safe).
    Rebuilt parents (whose children became temps) only become countable in
    the next pass, hence the fixpoint loop.
    """
    current = program
    if _has_loop(list(program.bindings)):
        # CSE hoists temps before their first use, which is unsafe across
        # loop boundaries (zero-trip loops would skip the temp). Loops are
        # rare; skip CSE entirely for loop-containing programs (plan §8:
        # CSE is not needed for correctness).
        return current
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
