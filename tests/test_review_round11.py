"""Regression tests for the adversarial-review findings (round 11)."""

from __future__ import annotations

import numpy as np
from support import make_fn

from vectorizer import vectorize

# 1: boolean min/max results participate in boolean-arithmetic conversion


def test_bool_minmax_arithmetic_exact() -> None:
    # min of two bool expressions keeps kind 'bool', so a + a intifies
    vec = vectorize(make_fn("    a = min(x > 0, x > 1)\n    return a + a"))
    assert vec(np.asarray([2]))[0] == 2


def test_bool_minmax_mixed_arithmetic() -> None:
    vec = vectorize(make_fn("    return max(x > 0, x > 1) + (x > 0)"))
    assert vec(np.asarray([2]))[0] == 2


def test_bool_minmax_bitwise_downstream() -> None:
    vec = vectorize(make_fn("    a = max(x > 0, x > 1)\n    return (a + a) & 1"))
    assert vec(np.asarray([2]))[0] == 0


# 2: boolean operands in the min/max helper work on strict backends


def test_bool_minmax_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return min(x > 0, x > 1)"))
    got = vec(xps.asarray([2, 0]))
    assert list(np.asarray(got, dtype=bool)) == [True, False]


def test_bool_minmax_strict_max() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return max(x > 0, x > 1)"))
    got = vec(xps.asarray([2, 0]))
    assert list(np.asarray(got, dtype=bool)) == [True, False]


def test_uint64_bool_mix_minmax() -> None:
    # the uint64 paths' signed-clamp comparisons must never see bools
    vec = vectorize(make_fn("    return min(x, y)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64), np.asarray([True]))
    assert int(got[0]) == 1
    vec = vectorize(make_fn("    return max(x, y)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64), np.asarray([True]))
    assert int(got[0]) == 2**63 + 1


def test_bool_minmax_with_literals() -> None:
    vec = vectorize(make_fn("    return max(x > 0, False)"))
    assert list(np.asarray(vec(np.asarray([2, -1])), dtype=bool)) == [True, False]
