"""The vectorizer IR: node dataclasses, SSA naming, and the tree algebra."""

from __future__ import annotations

from .nodes import (
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
from .ssa import RESERVED_NAMES, SSAEnv, generated_name
from .walk import children, rewrite

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
