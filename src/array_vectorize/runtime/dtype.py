# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Dtype analysis for the call-time helpers (from _runtime).

Classifies values/dtypes for promotion decisions: width extraction,
exact scalar-fit checks, and the min/max bound strategy. This code is
injected into generated modules and executes on the user's backend.
"""

from __future__ import annotations

import struct
from typing import Any

__all__ = ["_describe_dtype", "_dtype_bits", "_fits_dtype", "_minmax_bound_kind"]


def _describe_dtype(value: Any) -> tuple[bool, int]:
    """Classify a value for promotion: (is_float, width).

    Arrays contribute their dtype; float literals count as float64; int
    literals count as the minimal dtype that fits. Unsigned ints do not
    fit the same-width signed dtype (uint8 + int8 promotes to int16), and
    uint64 fits no signed int at all, so it counts as float64.
    """
    if hasattr(value, "dtype"):
        name = str(value.dtype)
    elif isinstance(value, float):
        name = "float64"
    else:
        v = int(value)
        if -(2**7) <= v < 2**7:
            name = "int8"
        elif -(2**15) <= v < 2**15:
            name = "int16"
        elif -(2**31) <= v < 2**31:
            name = "int32"
        else:
            name = "int64"
    is_float = "float" in name or "complex" in name
    digits = "".join(d for d in name if d.isdigit())
    width = int(digits) if digits else (64 if is_float else 8)
    if "uint" in name and not is_float:
        if width >= 64:
            return True, 64  # uint64: only float64 can hold it
        width = min(width * 2, 64)
    return is_float, width


def _fits_dtype(value: Any, dtype: Any) -> bool:
    """True when a raw Python scalar is EXACTLY representable in ``dtype``.

    Floats are checked by an IEEE round trip at the dtype's width (float32
    cannot hold 16777217); ints by range; bools only in bool dtypes.
    """
    name = str(dtype)
    if "bool" in name:
        return isinstance(value, bool)
    digits = "".join(d for d in name if d.isdigit())
    if "float" in name:
        code = {16: "e", 32: "f", 64: "d"}.get(int(digits) if digits else 64)
        if code is None:
            return False  # exotic float width: promote instead of risking rounding
        try:
            packed = struct.unpack("<" + code, struct.pack("<" + code, value))[0]
            return bool(packed == value)
        except (OverflowError, ValueError, struct.error):
            return False
    try:
        v = int(value)
    except (TypeError, ValueError, OverflowError):
        return False
    if v != value:  # e.g. a float literal for an int dtype
        return False
    bits = int(digits) if digits else 64
    if "uint" in name:
        return bool(0 <= v < 2**bits)
    return bool(-(2 ** (bits - 1)) <= v < 2 ** (bits - 1))


def _dtype_bits(dtype: Any) -> int:
    digits = "".join(d for d in str(dtype) if d.isdigit())
    return int(digits) if digits else 64


def _minmax_bound_kind(lit: Any, is_min: Any, dtype: Any) -> str:
    """``"fit"``, ``"clamp"``, or ``"promote"`` for a literal min/max bound.

    A bound that cannot win is CLAMPED into an unsigned array dtype
    instead of forcing promotion: max(unsigned, negative) never selects
    the bound, and min(unsigned, >= 2**bits) never selects it either —
    replacing it with the dtype's own extreme preserves the exact array
    values (float64 cannot hold every uint64).
    """
    if _fits_dtype(lit, dtype):
        return "fit"
    name = str(dtype)
    # max with a negative bound, or min with a too-large bound
    if (
        "uint" in name
        and isinstance(lit, int)
        and not isinstance(lit, bool)
        and ((not is_min and lit < 0) or (is_min and lit >= 2 ** _dtype_bits(dtype)))
    ):
        return "clamp"
    return "promote"
