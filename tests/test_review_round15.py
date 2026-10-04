"""Regression tests for the adversarial-review findings (round 15)."""

from __future__ import annotations

import numpy as np
from support import make_fn

from vectorizer import vectorize

# 1: uint64 arithmetic with negative literals stays exact (modular)


def test_uint64_negative_literal_arithmetic_exact() -> None:
    # (2**63 + 3) + (-2) = 2**63 + 1: fits uint64; modular uint64
    # arithmetic is exact instead of rounding through float64
    vec = vectorize(make_fn("    a = max(x, True)\n    return a + -2"))
    got = vec(np.asarray([2**63 + 3], dtype=np.uint64))
    assert int(got[0]) == 9223372036854775809


# 2: bitwise expressions preserve maybe-boolean tracking


def test_bitwise_maybe_bool_arithmetic() -> None:
    vec = vectorize(make_fn("    a = min(x, True)\n    b = a & True\n    return b + b"))
    assert list(np.asarray(vec(np.asarray([True, False])))) == [2, 0]


def test_bitwise_maybe_bool_or() -> None:
    vec = vectorize(make_fn("    a = min(x, True)\n    b = a | False\n    return b + b"))
    assert list(np.asarray(vec(np.asarray([True, False])))) == [2, 0]


def test_bitwise_maybe_bool_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = min(x, True)\n    b = a & True\n    return b + b"))
    got = vec(xps.asarray([True, False]))
    assert list(map(int, np.asarray(got))) == [2, 0]


def test_arithmetic_maybe_bool_result_not_maybe_bool() -> None:
    # a + 1 intifies inside, so the result is numeric and needs no
    # second conversion (values stay exact)
    vec = vectorize(make_fn("    a = min(x, True)\n    b = a + 1\n    return b + b"))
    assert list(np.asarray(vec(np.asarray([True, False])))) == [4, 2]
