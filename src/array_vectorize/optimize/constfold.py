"""Constant folding over the IR (plan §8)."""

from __future__ import annotations

import math
import operator
from typing import Any

from ..ir import (
    Binding,
    BinOp,
    Call,
    Compare,
    DType,
    Kind,
    Literal,
    Loop,
    Node,
    Program,
    Stmt,
    UnaryOp,
)
from ..ir.walk import rewrite

__all__ = ["const_fold"]

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1

#: pow is never folded (backend pow semantics on edge values differ);
#: div/mod/floordiv by literal zero are skipped (runtime inf/NaN, plan D3).
_NO_FOLD_OPS = frozenset({"pow"})
_ZERO_RISK_OPS = frozenset({"div", "floordiv", "mod"})


def _int64_ok(value: int) -> bool:
    return _INT64_MIN <= value <= _INT64_MAX


def _literal_kind(value: object) -> Kind:
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
        match op:
            case "add":
                result: object = a + b
            case "sub":
                result = a - b
            case "mul":
                result = a * b
            case "div":
                result = a / b
            case "floordiv":
                result = a // b
            case "mod":
                result = a % b
            case "and":
                result = a & b
            case "or":
                result = a | b
            case "xor":
                result = a ^ b
            case "lshift":
                result = a << b
            case "rshift":
                result = a >> b
            case _:
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
        match op:
            case "neg":
                result: Any = -v
            case "pos":
                result = +v
            case "invert":
                result = ~v
            case _:
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
    node = rewrite(node, _const_fold_expr)
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
