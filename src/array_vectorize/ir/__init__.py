# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""The array_vectorize IR: node dataclasses, SSA naming, and the tree algebra."""

from __future__ import annotations

from array_vectorize.ir.nodes import Binding
from array_vectorize.ir.nodes import BinOp
from array_vectorize.ir.nodes import Call
from array_vectorize.ir.nodes import Compare
from array_vectorize.ir.nodes import DType
from array_vectorize.ir.nodes import FuncCall
from array_vectorize.ir.nodes import is_bool
from array_vectorize.ir.nodes import Kind
from array_vectorize.ir.nodes import Literal
from array_vectorize.ir.nodes import Logical
from array_vectorize.ir.nodes import Loop
from array_vectorize.ir.nodes import Node
from array_vectorize.ir.nodes import Program
from array_vectorize.ir.nodes import Ref
from array_vectorize.ir.nodes import Stmt
from array_vectorize.ir.nodes import UnaryOp
from array_vectorize.ir.nodes import Where
from array_vectorize.ir.ssa import generated_name
from array_vectorize.ir.ssa import RESERVED_NAMES
from array_vectorize.ir.ssa import SSAEnv
from array_vectorize.ir.walk import children
from array_vectorize.ir.walk import rewrite

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
