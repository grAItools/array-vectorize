"""Regression tests for the adversarial-review findings (round 12)."""

from __future__ import annotations

import numpy as np
from support import make_fn

from vectorizer import vectorize

# 1: boolean/int min/max selections get int64 headroom downstream


def test_bool_int_minmax_no_int8_overflow() -> None:
    # min(bool, 1) selects in int64: a * 100 * 2 must not wrap int8
    vec = vectorize(make_fn("    a = min(x > 0, 1)\n    return a * 100 * 2"))
    assert vec(np.asarray([2]))[0] == 200


def test_bool_int_minmax_large_literal() -> None:
    vec = vectorize(make_fn("    a = max(x > 0, 300)\n    return a * 100"))
    assert vec(np.asarray([2]))[0] == 30000


# 2: boolean-only min/max restores boolean results


def test_bool_only_minmax_bitwise_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return min(x > 0, x > 1) & (x > 0)"))
    got = vec(xps.asarray([2]))
    assert bool(np.asarray(got).reshape(-1)[0])


def test_bool_only_minmax_bitwise_numpy() -> None:
    vec = vectorize(make_fn("    return min(x > 0, x > 1) & (x > 0)"))
    # scalar: f(2) = min(True, True) & True = True; f(0) = False & False
    assert list(np.asarray(vec(np.asarray([2, 0])), dtype=bool)) == [True, False]


def test_bool_only_minmax_or_bitwise() -> None:
    vec = vectorize(make_fn("    return max(x > 0, x > 1) | (x > 5)"))
    assert list(np.asarray(vec(np.asarray([2, 0])), dtype=bool)) == [True, False]


def test_bool_only_minmax_still_arithmetics_exact() -> None:
    # bool result + bool result still intifies through kind inference
    vec = vectorize(make_fn("    a = min(x > 0, x > 1)\n    return a + a"))
    assert vec(np.asarray([2]))[0] == 2


# mixed bool/float selections stay float


def test_bool_float_minmax() -> None:
    vec = vectorize(make_fn("    return min(x > 0, 1.5)"))
    assert vec(np.asarray([2]))[0] == 1.0
