"""Optimizer passes: const-fold, CSE, DCE (plan §8)."""

from __future__ import annotations

import math
import operator
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


#: Array API functions that are pure and safe to fold over literal args,
#: mapped to their Python equivalents. Keeps plain scalars out of generated
#: xp.* calls (strict backends require arrays) and preserves semantics.
_FOLDABLE_CALLS: dict[str, Any] = {
    "sqrt": math.sqrt,
    "exp": math.exp,
    "expm1": math.expm1,
    "log": math.log,
    "log1p": math.log1p,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "asinh": math.asinh,
    "acosh": math.acosh,
    "atanh": math.atanh,
    "pow": pow,
    "floor": math.floor,
    "ceil": math.ceil,
    "hypot": math.hypot,
    "copysign": math.copysign,
    "abs": abs,
    "round": round,
    "minimum": min,
    "maximum": max,
    "isfinite": math.isfinite,
    "isnan": math.isnan,
    "isinf": math.isinf,
    "logical_not": lambda v: not v,
}

_CMPOP_PY: dict[str, Any] = {
    "eq": operator.eq,
    "ne": operator.ne,
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
}


def _fold_compare(op: str, left: Literal, right: Literal) -> Literal | None:
    try:
        result = _CMPOP_PY[op](left.value, right.value)
    except (ArithmeticError, ValueError, TypeError):
        return None
    return Literal(bool(result), "bool")


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
    if isinstance(node, Compare):
        node = Compare(node.op, _const_fold_expr(node.left), _const_fold_expr(node.right))
        if isinstance(node.left, Literal) and isinstance(node.right, Literal):
            folded = _fold_compare(node.op, node.left, node.right)
            if folded is not None:
                return folded
        return node
    node = _rewrite(node, _const_fold_expr)
    if isinstance(node, Call) and node.fn in _FOLDABLE_CALLS and node.args:
        values = [a.value for a in node.args if isinstance(a, Literal)]
        if len(values) != len(node.args):
            return node
        try:
            value = _FOLDABLE_CALLS[node.fn](*values)
        except (ValueError, ArithmeticError, OverflowError, TypeError):
            pass
        else:
            if isinstance(value, bool):
                return Literal(value, "bool")
            if isinstance(value, int) and _int64_ok(value):
                return Literal(value, "int")
            if isinstance(value, float):
                return Literal(value, "float")
    # constant casts: int(3.14) -> 3 (also keeps plain-float astype out of
    # generated code, where xp.astype would fail on non-arrays)
    if (
        isinstance(node, Call)
        and node.fn == "astype"
        and len(node.args) == 2
        and isinstance(node.args[0], Literal)
        and isinstance(node.args[1], DType)
    ):
        value, dtype = node.args[0].value, node.args[1].name
        try:
            if dtype == "bool":
                return Literal(bool(value), "bool")
            if dtype == "float64":
                return Literal(float(value), "float")
            if _int64_ok(int(value)):
                return Literal(int(value), "int")
        except (ValueError, OverflowError):
            pass
        # int(inf)/int(nan) cannot fold (scalar Python raises Overflow/
        # ValueError): keep the cast but wrap the literal so xp.astype gets
        # an array, not a plain float (backend-defined result, plan D3)
        return Call("astype", (Call("asarray", (node.args[0],)), node.args[1]))
    return node


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


def _mark_live_stmt(stmt: Stmt, live: set[str]) -> Stmt | None:
    """Mark liveness from a statement; returns the kept statement or None."""
    if isinstance(stmt, Binding):
        if stmt.name in live:
            _mark_live(stmt.expr, live)
            return stmt
        return None
    # Loop: recompute liveness from the FULL body in reverse order until a
    # fixed point: a statement can be live only through a use appearing
    # EARLIER in the body (loop-carried feedback), which a single reverse
    # pass misses. Liveness only grows, so this terminates.
    kept: list[Stmt] = []
    while True:
        before_live = set(live)
        new_kept: list[Stmt] = []
        for inner in reversed(stmt.body):
            kept_inner = _mark_live_stmt(inner, live)
            if kept_inner is not None:
                new_kept.append(kept_inner)
        new_kept.reverse()
        kept = new_kept
        if live == before_live:
            break
    kept_any = len(kept) > 0
    _mark_live(stmt.start, live)
    _mark_live(stmt.stop, live)
    _mark_live(stmt.step, live)
    if kept_any or stmt.var in live:
        return Loop(stmt.var, stmt.start, stmt.stop, stmt.step, tuple(kept))
    return None


def dce(program: Program) -> Program:
    live: set[str] = set()
    _mark_live(program.result, live)
    kept: list[Stmt] = []
    # single reverse pass suffices: SSA uses only earlier bindings
    for stmt in reversed(program.bindings):
        kept_stmt = _mark_live_stmt(stmt, live)
        if kept_stmt is not None:
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
        for stmt in current.bindings:
            assert isinstance(stmt, Binding)  # CSE skips loop programs
            _count_eligible(stmt.expr, counter)
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

        rewritten: list[Stmt] = []
        for stmt in current.bindings:
            assert isinstance(stmt, Binding)  # CSE skips loop programs
            start = len(pending)
            expr = rewrite(stmt.expr)
            rewritten.extend(pending[start:])
            rewritten.append(Binding(stmt.name, expr))
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


# ------------------------------------------------------- protect_domains

#: Partial functions: (lo, lo_open, hi, hi_open, safe_lo, safe_hi).
#: Protection clamps a partial call's argument ONLY on lanes that are dead
#: (discarded by the enclosing where-select) AND out of domain; live lanes
#: are never touched, so results never change (plan D1, M4). Open bounds
#: (poles such as log(0)) clamp dead lanes to an interior constant, so no
#: warnings are produced and float32 is safe.
_PARTIAL_DOMAINS: dict[str, tuple[float, bool, float | None, bool, float, float | None]] = {
    "sqrt": (0.0, False, None, False, 0.0, None),
    "log": (0.0, True, None, False, 1.0, None),
    "log2": (0.0, True, None, False, 1.0, None),
    "log10": (0.0, True, None, False, 1.0, None),
    "log1p": (-1.0, True, None, False, 0.0, None),
    "asin": (-1.0, False, 1.0, False, -1.0, 1.0),
    "acos": (-1.0, False, 1.0, False, -1.0, 1.0),
    "acosh": (1.0, False, None, False, 1.0, None),
    "atanh": (-1.0, True, 1.0, True, 0.0, 0.0),
}


def _and_ctx(ctx: Node | None, cond: Node) -> Node:
    return cond if ctx is None else Logical("and", (ctx, cond))


def _clamp_dead(arg: Node, dead: Node, spec: tuple) -> Node:
    """Clamp ``arg`` to the domain, but only where ``dead`` holds."""
    lo, lo_open, hi, hi_open, safe_lo, safe_hi = spec
    clamped: Node = arg
    if hi is not None:
        op = "ge" if hi_open else "gt"
        clamped = Where(
            Logical("and", (dead, Compare(op, arg, Literal(hi, "float")))),
            Literal(safe_hi, "float"),
            clamped,
        )
    op = "le" if lo_open else "lt"
    clamped = Where(
        Logical("and", (dead, Compare(op, arg, Literal(lo, "float")))),
        Literal(safe_lo, "float"),
        clamped,
    )
    return clamped


def protect_domains(program: Program) -> Program:
    """Clamp partial-function arguments on dead lanes only (plan D1, M4).

    A single reverse pass propagates liveness conditions: each binding's
    value is live under the disjunction of the conditions governing its uses
    (where-branch guards, conjunction-nested). A partial call's argument is
    clamped only where the value is dead AND out of domain, so live results
    never change and dead lanes never warn. Loop bodies are left unprotected
    (conservative) and their reads mark bindings fully live.
    """
    live: dict[str, Node | None] = {}
    _unrecorded = object()

    def or_live(name: str, ctx: Node | None) -> None:
        cur = live.get(name, _unrecorded)  # type: ignore[arg-type]
        if cur is _unrecorded:
            live[name] = ctx
        elif cur is None or ctx is None:
            live[name] = None
        else:
            live[name] = Logical("or", (cur, ctx))

    def record_uses(node: Node, ctx: Node | None) -> None:
        if isinstance(node, Ref):
            or_live(node.name, ctx)
        elif isinstance(node, Where):
            record_uses(node.cond, ctx)
            record_uses(node.then, _and_ctx(ctx, node.cond))
            record_uses(node.other, _and_ctx(ctx, UnaryOp("not", node.cond)))
        else:
            for child in _children(node):
                record_uses(child, ctx)

    def rewrite(node: Node, ctx: Node | None) -> Node:
        if isinstance(node, Ref):
            return node
        if isinstance(node, Where):
            cond = rewrite(node.cond, ctx)
            then = rewrite(node.then, _and_ctx(ctx, cond))
            other = rewrite(node.other, _and_ctx(ctx, UnaryOp("not", cond)))
            return Where(cond, then, other)
        if isinstance(node, Call) and node.fn in _PARTIAL_DOMAINS:
            args = [rewrite(a, ctx) for a in node.args]
            if ctx is not None:
                dead = UnaryOp("not", ctx)
                spec = _PARTIAL_DOMAINS[node.fn]
                args = [_clamp_dead(a, dead, spec) for a in args]
            return Call(node.fn, tuple(args))
        return _rewrite(node, lambda n: rewrite(n, ctx))

    def mark_fully_live(node: Node) -> None:
        if isinstance(node, Ref):
            live[node.name] = None
        else:
            for child in _children(node):
                mark_fully_live(child)

    def rewrite_stmt(stmt: Stmt) -> Stmt:
        if isinstance(stmt, Binding):
            ctx = live.pop(stmt.name, None)
            expr = rewrite(stmt.expr, ctx)
            record_uses(expr, ctx)
            return Binding(stmt.name, expr)
        assert isinstance(stmt, Loop)
        # conservative: loop bodies are not rewritten; every name they read
        # is marked fully live (later iterations may consume any binding)
        mark_fully_live(stmt.start)
        mark_fully_live(stmt.stop)
        mark_fully_live(stmt.step)
        for inner in stmt.body:
            if isinstance(inner, Binding):
                mark_fully_live(inner.expr)
        return stmt

    new_result = rewrite(program.result, None)
    record_uses(new_result, None)
    rewritten = [rewrite_stmt(stmt) for stmt in reversed(program.bindings)]
    rewritten.reverse()
    return Program(program.params, tuple(rewritten), new_result)
