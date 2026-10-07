# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""CALL-TIME runtime library injected into generated modules (from _runtime).

Executes on the user's backend when a vectorized function is called.
This package is a leaf: it imports nothing from the rest of array_vectorize,
only the stdlib. Generation-time loading lives in array_vectorize.emit.
"""

from __future__ import annotations

from array_vectorize.runtime.arith import _vec_arith
from array_vectorize.runtime.arith import ArithOp
from array_vectorize.runtime.dtype import _fits_dtype
from array_vectorize.runtime.minmax import _vec_minmax
from array_vectorize.runtime.registry import RUNTIME_HELPERS

__all__ = ["RUNTIME_HELPERS", "ArithOp", "_fits_dtype", "_vec_arith", "_vec_minmax"]
