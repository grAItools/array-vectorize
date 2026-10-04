"""Regression tests for the adversarial-review findings (round 23)."""

from __future__ import annotations

import numpy as np
from support import make_fn

from vectorizer import vectorize

# result-aware subtraction with non-negative signed arrays


def test_int64_minus_uint64_huge_exact() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return y - a", defaults="x, y"))
    got = vec(np.asarray([2**63], dtype=np.uint64), np.asarray([2**63 - 1], dtype=np.int64))
    assert int(got[0]) == -1


def test_int64_minus_uint64_huge_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = max(x, True)\n    return y - a", defaults="x, y"))
    got = vec(xps.asarray([2**63], dtype=xps.uint64), xps.asarray([2**63 - 1], dtype=xps.int64))
    assert int(got[0]) == -1


def test_sub_result_int64_min_exact() -> None:
    # -(2**63) is exactly int64-min: the mod-2**64 subtraction yields it
    vec = vectorize(make_fn("    a = max(x, True)\n    return y - a", defaults="x, y"))
    got = vec(np.asarray([2**63], dtype=np.uint64), np.asarray([0], dtype=np.int64))
    assert int(got[0]) == -(2**63)


def test_uint64_minus_signed_negative_result_exact() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return a - y", defaults="x, y"))
    got = vec(np.asarray([5], dtype=np.uint64), np.asarray([7], dtype=np.int64))
    assert int(got[0]) == -2


def test_sub_nonnegative_result_keeps_uint64() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return a - y", defaults="x, y"))
    got = vec(np.asarray([2**63 + 5], dtype=np.uint64), np.asarray([3], dtype=np.int64))
    assert int(got[0]) == 2**63 + 2


def test_sub_mixed_magnitude_fallback_documented() -> None:
    # per-lane differences spanning u64-only positives and int64
    # negatives fit no single dtype: float64 approximation
    vec = vectorize(make_fn("    a = max(x, True)\n    return a - y", defaults="x, y"))
    got = vec(np.asarray([2**63 + 5, 1], dtype=np.uint64), np.asarray([3, 4], dtype=np.int64))
    assert np.isclose(float(got[0]), 2**63 + 2)
    assert np.isclose(float(got[1]), -3)
