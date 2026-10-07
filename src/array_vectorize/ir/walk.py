# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Generic IR tree algebra: child enumeration and bottom-up rebuilding.

``children``/``rewrite`` moved from ``_optimize`` (restructure phase 2): the
IR owns its own algebra, not a consumer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import assert_never

from array_vectorize.ir import nodes

__all__ = ["children", "rewrite"]


def rewrite(node: nodes.Node, fn: Callable[[nodes.Node], nodes.Node]) -> nodes.Node:
    """Rebuild ``node`` with ``fn`` applied to each rewritten child."""
    match node:
        case nodes.Literal() | nodes.Ref() | nodes.DType():
            return node
        case nodes.BinOp():
            return nodes.BinOp(node.op, fn(node.left), fn(node.right))
        case nodes.UnaryOp():
            return nodes.UnaryOp(node.op, fn(node.operand))
        case nodes.Compare():
            return nodes.Compare(node.op, fn(node.left), fn(node.right))
        case nodes.Logical():
            return nodes.Logical(node.op, tuple(fn(p) for p in node.parts))
        case nodes.Where():
            return nodes.Where(fn(node.cond), fn(node.then), fn(node.other))
        case nodes.Call():
            return nodes.Call(node.fn, tuple(fn(a) for a in node.args))
        case nodes.FuncCall():
            return nodes.FuncCall(node.fn, tuple(fn(a) for a in node.args))
        case _:
            assert_never(node)


def children(node: nodes.Node) -> tuple[nodes.Node, ...]:
    """The direct operand sub-nodes of ``node`` (leaves have none)."""
    match node:
        case nodes.Literal() | nodes.Ref() | nodes.DType():
            return ()
        case nodes.BinOp():
            return (node.left, node.right)
        case nodes.UnaryOp():
            return (node.operand,)
        case nodes.Compare():
            return (node.left, node.right)
        case nodes.Logical():
            return node.parts
        case nodes.Where():
            return (node.cond, node.then, node.other)
        case nodes.Call():
            return node.args
        case nodes.FuncCall():
            return node.args
        case _:
            assert_never(node)
