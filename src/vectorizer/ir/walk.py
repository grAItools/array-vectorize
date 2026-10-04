"""Generic IR tree algebra: child enumeration and bottom-up rebuilding.

``children``/``rewrite`` moved from ``_optimize`` (restructure phase 2): the
IR owns its own algebra, not a consumer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import assert_never

from .nodes import (
    BinOp,
    Call,
    Compare,
    DType,
    FuncCall,
    Literal,
    Logical,
    Node,
    Ref,
    UnaryOp,
    Where,
)

__all__ = ["children", "rewrite"]


def rewrite(node: Node, fn: Callable[[Node], Node]) -> Node:
    """Rebuild ``node`` with ``fn`` applied to each rewritten child."""
    match node:
        case Literal() | Ref() | DType():
            return node
        case BinOp():
            return BinOp(node.op, fn(node.left), fn(node.right))
        case UnaryOp():
            return UnaryOp(node.op, fn(node.operand))
        case Compare():
            return Compare(node.op, fn(node.left), fn(node.right))
        case Logical():
            return Logical(node.op, tuple(fn(p) for p in node.parts))
        case Where():
            return Where(fn(node.cond), fn(node.then), fn(node.other))
        case Call():
            return Call(node.fn, tuple(fn(a) for a in node.args))
        case FuncCall():
            return FuncCall(node.fn, tuple(fn(a) for a in node.args))
        case _:
            assert_never(node)


def children(node: Node) -> tuple[Node, ...]:
    match node:
        case Literal() | Ref() | DType():
            return ()
        case BinOp():
            return (node.left, node.right)
        case UnaryOp():
            return (node.operand,)
        case Compare():
            return (node.left, node.right)
        case Logical():
            return node.parts
        case Where():
            return (node.cond, node.then, node.other)
        case Call():
            return node.args
        case FuncCall():
            return node.args
        case _:
            assert_never(node)
