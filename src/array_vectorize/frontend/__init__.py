"""Frontend: understand the scalar function (extract, validate, tables).

NOTE: the ``validate`` function is deliberately NOT re-exported here —
binding it would clobber the ``frontend.validate`` submodule attribute.
Import it as ``from array_vectorize.frontend.validate import validate``.
"""

from __future__ import annotations

from .extract import extract_function
from .info import FunctionInfo, Param, ParamKind
from .lambda_id import _find_target
from .tables import (
    BUILTIN_CASTS,
    BUILTIN_FOLDS,
    BUILTIN_UNARY,
    MATH_CONSTS,
    MATH_FUNCS,
    MATH_SPECIAL,
)

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
