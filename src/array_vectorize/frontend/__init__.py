# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Frontend: understand the scalar function (extract, validate, tables).

NOTE: the ``validate`` function is deliberately NOT re-exported here —
binding it would clobber the ``frontend.validate`` submodule attribute.
Import it as ``from array_vectorize.frontend.validate import validate``.
"""

from __future__ import annotations

from array_vectorize.frontend.extract import extract_function
from array_vectorize.frontend.info import FunctionInfo
from array_vectorize.frontend.info import Param
from array_vectorize.frontend.info import ParamKind
from array_vectorize.frontend.lambda_id import _find_target
from array_vectorize.frontend.tables import BUILTIN_CASTS
from array_vectorize.frontend.tables import BUILTIN_FOLDS
from array_vectorize.frontend.tables import BUILTIN_UNARY
from array_vectorize.frontend.tables import MATH_CONSTS
from array_vectorize.frontend.tables import MATH_FUNCS
from array_vectorize.frontend.tables import MATH_SPECIAL

__all__ = [
    "BUILTIN_CASTS",
    "BUILTIN_FOLDS",
    "BUILTIN_UNARY",
    "MATH_CONSTS",
    "MATH_FUNCS",
    "MATH_SPECIAL",
    "FunctionInfo",
    "Param",
    "ParamKind",
    "_find_target",
    "extract_function",
]
