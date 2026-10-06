"""The array_vectorize IR: node dataclasses, SSA naming, and the tree algebra."""

from __future__ import annotations

from array_vectorize.ir.nodes import (
    Binding,
    BinOp,
    Call,
    Compare,
    DType,
    FuncCall,
    Kind,
    Literal,
    Logical,
    Loop,
    Node,
    Program,
    Ref,
    Stmt,
    UnaryOp,
    Where,
    is_bool,
)
from array_vectorize.ir.ssa import RESERVED_NAMES, SSAEnv, generated_name
from array_vectorize.ir.walk import children, rewrite

__all__ = [
    "RESERVED_NAMES",
    "BinOp",
    "Binding",
    "Call",
    "Compare",
    "DType",
    "FuncCall",
    "Kind",
    "Literal",
    "Logical",
    "Loop",
    "Node",
    "Program",
    "Ref",
    "SSAEnv",
    "Stmt",
    "UnaryOp",
    "Where",
    "children",
    "generated_name",
    "is_bool",
    "rewrite",
]
