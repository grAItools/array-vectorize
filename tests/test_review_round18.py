"""Regression tests for the adversarial-review findings (round 18)."""

from __future__ import annotations

import numpy as np
from support import make_fn

from vectorizer import vectorize

# per-lane exactness: one huge uint64 lane must not degrade the others


def test_uint64_mod_negative_mixed_lanes_exact() -> None:
    # both remainders fit int64: computed exactly per lane from uint64
    # magnitudes, regardless of input magnitude
    vec = vectorize(make_fn("    return max(x, True) % -2"))
    got = vec(np.asarray([2**53 + 1, 2**63 + 1], dtype=np.uint64))
    assert list(map(int, np.asarray(got))) == [-1, -1]


def test_uint64_floordiv_negative_mixed_lanes_exact() -> None:
    vec = vectorize(make_fn("    return max(x, True) // -2"))
    got = vec(np.asarray([2**53 + 1, 2**63 + 1], dtype=np.uint64))
    assert list(map(int, np.asarray(got))) == [-4503599627370497, -4611686018427387905]


def test_uint64_negate_mixed_lanes_best_effort() -> None:
    # -(2**63 + 1) is unrepresentable in int64: the batch falls back to
    # float64 (documented); the small lane approximates
    vec = vectorize(make_fn("    return -max(x, True)"))
    got = vec(np.asarray([2**53 + 1, 2**63 + 1], dtype=np.uint64))
    assert np.isclose(float(np.asarray(got).reshape(-1)[0]), -9007199254740993)


def test_uint64_negative_literal_mod_array_mixed() -> None:
    # literal-left remainder: (v - |d| mod v) mod v, exact in uint64
    vec = vectorize(make_fn("    return -2 % max(x, True)"))
    got = vec(np.asarray([2**53 + 1, 2**63 + 1], dtype=np.uint64))
    assert list(map(int, np.asarray(got))) == [2**53 - 1, 2**63 - 1]


def test_uint64_negative_literal_floordiv_array() -> None:
    # (-d) // v == -ceil(d / v), exact in int64
    vec = vectorize(make_fn("    return -5 // max(x, True)"))
    got = vec(np.asarray([2], dtype=np.uint64))
    assert int(got[0]) == -3  # -5 // 2 == -3


def test_uint64_add_negative_literal_modular_still_exact() -> None:
    vec = vectorize(make_fn("    return max(x, True) + -2"))
    got = vec(np.asarray([2**63 + 3], dtype=np.uint64))
    assert int(got[0]) == 9223372036854775809


def test_unary_negate_maybe_bool_after_restructure() -> None:
    vec = vectorize(make_fn("    a = min(x, True)\n    return -a"))
    assert list(np.asarray(vec(np.asarray([True, False])))) == [-1, 0]


def test_truediv_bool_strict_after_restructure() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return min(x, True) / 2"))
    got = vec(xps.asarray([True, False]))
    assert list(map(float, np.asarray(got))) == [0.5, 0.0]
