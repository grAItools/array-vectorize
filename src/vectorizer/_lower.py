"""Compat shim: canonical home is vectorizer.lower (restructure phase 4a)."""

from __future__ import annotations

from .lower import HelperVectorizer, LoweredFunction, _Lowerer, lower_function
from .lower.expressions import _BINOPS, _CMPOPS, _UNARY

__all__ = [
    "_BINOPS",
    "_CMPOPS",
    "_UNARY",
    "HelperVectorizer",
    "LoweredFunction",
    "_Lowerer",
    "lower_function",
]
