"""Compat shim: canonical home is vectorizer.frontend.tables (restructure phase 2)."""

from .frontend.tables import (
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
]
