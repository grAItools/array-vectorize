"""Compat shim: canonical homes are vectorizer.emit (generation-time
compile_vectorized) and vectorizer.runtime (call-time helpers) —
restructure phase 3."""

from __future__ import annotations

from .emit import compile_vectorized
from .runtime.arith import (
    _ARITH_FNS,
    _ARITH_OPS,
    ArithOp,
    _is_u64,
    _to_u64,
    _u64_sub_exact,
    _vec_arith,
    u64_dividend,
)
from .runtime.dtype import _describe_dtype, _dtype_bits, _fits_dtype, _minmax_bound_kind
from .runtime.minmax import _vec_minmax

__all__ = [
    "_ARITH_FNS",
    "_ARITH_OPS",
    "ArithOp",
    "_describe_dtype",
    "_dtype_bits",
    "_fits_dtype",
    "_is_u64",
    "_minmax_bound_kind",
    "_to_u64",
    "_u64_sub_exact",
    "_vec_arith",
    "_vec_minmax",
    "compile_vectorized",
    "u64_dividend",
]
