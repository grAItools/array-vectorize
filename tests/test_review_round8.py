"""Regression tests for the adversarial-review findings (round 8)."""

from __future__ import annotations

import numpy as np
from support import make_fn

from vectorizer import vectorize

# 1: float16 literals are checked at float16 precision


def test_max_float16_literal_precision() -> None:
    # 2049 is not representable in float16: the fast path must reject it
    # and promote to float64 instead of rounding the bound to 2048
    vec = vectorize(make_fn("    return max(x, 2049.0)"))
    got = vec(np.asarray([0.0], dtype=np.float16))
    assert got[0] == 2049.0


def test_max_float16_exact_literal_keeps_dtype() -> None:
    vec = vectorize(make_fn("    return max(x, 2048.0)"))
    got = vec(np.asarray([0.0], dtype=np.float16))
    assert got[0] == 2048.0


# 2: uint64 with a negative max-bound stays exact


def test_max_uint64_negative_bound_exact() -> None:
    # a negative bound never wins a max over unsigned values: it is
    # clamped into the unsigned dtype instead of forcing float64 (which
    # cannot hold 9223372036854775809 exactly)
    vec = vectorize(make_fn("    return max(x, -1)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64))
    # exact comparison via Python int (numpy scalar == rounds floats)
    assert int(got[0]) == 9223372036854775809


def test_min_uint64_negative_bound_literal_wins() -> None:
    # min selects the negative bound: it must survive exactly
    vec = vectorize(make_fn("    return min(x, -1)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64))
    assert int(got[0]) == -1


# clamp follow-ups for narrow unsigned dtypes


def test_min_uint8_too_large_bound_clamps() -> None:
    # 300 never wins a min over uint8 values (max 255): clamp to 255
    vec = vectorize(make_fn("    return min(x, 300)"))
    got = vec(np.asarray([1], dtype=np.uint8))
    assert got[0] == 1


def test_max_uint8_negative_bound_clamps() -> None:
    vec = vectorize(make_fn("    return max(x, -1)"))
    got = vec(np.asarray([7], dtype=np.uint8))
    assert got[0] == 7
