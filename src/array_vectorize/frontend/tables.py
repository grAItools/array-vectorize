# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Scalar-name -> Array API mapping tables.

All ``MATH_FUNCS`` targets must exist on every Array API backend; this is
enforced by tests against ``array_api_strict`` (the conformance reference).
"""

from __future__ import annotations

import math

__all__ = [
    "BUILTIN_CASTS",
    "BUILTIN_FOLDS",
    "BUILTIN_UNARY",
    "MATH_CONSTS",
    "MATH_FUNCS",
    "MATH_SPECIAL",
]

#: math.<name>(...) -> xp.<name>(...). Value: (xp_name, min_arity, max_arity).
#: math.hypot is variadic in Python but xp.hypot is binary: max_arity 2.
MATH_FUNCS: dict[str, tuple[str, int, int]] = {
    "sqrt": ("sqrt", 1, 1),
    "exp": ("exp", 1, 1),
    "expm1": ("expm1", 1, 1),
    "log": ("log", 1, 1),
    "log1p": ("log1p", 1, 1),
    "log2": ("log2", 1, 1),
    "log10": ("log10", 1, 1),
    "sin": ("sin", 1, 1),
    "cos": ("cos", 1, 1),
    "tan": ("tan", 1, 1),
    "asin": ("asin", 1, 1),
    "acos": ("acos", 1, 1),
    "atan": ("atan", 1, 1),
    "atan2": ("atan2", 2, 2),
    "sinh": ("sinh", 1, 1),
    "cosh": ("cosh", 1, 1),
    "tanh": ("tanh", 1, 1),
    "asinh": ("asinh", 1, 1),
    "acosh": ("acosh", 1, 1),
    "atanh": ("atanh", 1, 1),
    "pow": ("pow", 2, 2),
    "floor": ("floor", 1, 1),
    "ceil": ("ceil", 1, 1),
    "hypot": ("hypot", 2, 2),
    "copysign": ("copysign", 2, 2),
    "isnan": ("isnan", 1, 1),
    "isinf": ("isinf", 1, 1),
    "isfinite": ("isfinite", 1, 1),
}

#: math.<name> -> constant value (codegen: inf/nan emit xp.inf/xp.nan).
MATH_CONSTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    "nan": math.nan,
}

#: math functions with non-identity translations:
#: math.trunc(x) -> xp.astype(x, xp.int64) (trunc-toward-zero, like int()).
MATH_SPECIAL: dict[str, str] = {"trunc": "int64"}

#: Builtins translated to a single xp call: abs(x) -> xp.abs(x);
#: round(x) -> xp.round(x). round(x, n) is rejected (decimals kwarg is not in
#: the Array API standard - verified against array-api-strict).
BUILTIN_UNARY: dict[str, str] = {
    "abs": "abs",
    "round": "round",
}

#: Builtins lowered to a left-fold of an xp function, arity >= 2:
#: min(a, b, ...) -> xp.minimum(xp.minimum(a, b), c) ...
BUILTIN_FOLDS: dict[str, str] = {
    "min": "minimum",
    "max": "maximum",
}

#: Builtins lowered to casts: int(x) -> xp.astype(x, xp.int64), etc.
#: astype truncates toward zero, matching int().
BUILTIN_CASTS: dict[str, str] = {
    "int": "int64",
    "float": "float64",
    "bool": "bool",
}
