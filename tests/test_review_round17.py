"""Regression tests for the adversarial-review findings (round 17)."""

from __future__ import annotations

import numpy as np
from support import make_fn

from vectorizer import vectorize

# 1: large uint64 values use int64 when every value fits (exact)


def test_uint64_mod_negative_large_exact() -> None:
    # (2**53 + 1) % -2 == -1: fits int64; float64 would round the value
    vec = vectorize(make_fn("    a = max(x, True)\n    return a % -2"))
    got = vec(np.asarray([2**53 + 1], dtype=np.uint64))
    assert int(got[0]) == -1


def test_uint64_floordiv_negative_large_exact() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return a // -2"))
    got = vec(np.asarray([2**53 + 1], dtype=np.uint64))
    assert int(got[0]) == -4503599627370497


def test_uint64_negate_large_exact() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return -a"))
    got = vec(np.asarray([2**53 + 1], dtype=np.uint64))
    assert int(got[0]) == -9007199254740993


def test_uint64_negate_huge_best_effort_float() -> None:
    # values beyond int64: the result is unrepresentable exactly; the
    # cast falls back to float64 (documented best effort)
    vec = vectorize(make_fn("    a = max(x, True)\n    return -a"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64))
    assert np.isclose(float(got[0]), -(2.0**63 + 1))


# 2: true division selects floating-point operands


def test_truediv_bool_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return min(x, True) / 2"))
    got = vec(xps.asarray([True, False]))
    assert list(map(float, np.asarray(got))) == [0.5, 0.0]


def test_truediv_bool_numpy() -> None:
    vec = vectorize(make_fn("    return min(x, True) / 2"))
    got = vec(np.asarray([True, False]))
    assert list(np.asarray(got)) == [0.5, 0.0]
