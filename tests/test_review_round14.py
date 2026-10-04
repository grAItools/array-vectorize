"""Regression tests for the adversarial-review findings (round 14)."""

from __future__ import annotations

import numpy as np
from support import make_fn

from vectorizer import vectorize

# 1: branch merges keep maybe-boolean tracking


def test_branch_merged_maybe_bool_arithmetic() -> None:
    vec = vectorize(
        make_fn(
            "    if x == True:\n"
            "        a = min(x, True)\n"
            "    else:\n"
            "        a = max(x, False)\n"
            "    return a + a"
        )
    )
    assert list(np.asarray(vec(np.asarray([True, False])))) == [2, 0]


# 2: maybe-boolean conversion preserves uint64 values


def test_maybe_bool_uint64_arithmetic_exact() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return a + 0"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64))
    assert int(got[0]) == 9223372036854775809


def test_maybe_bool_uint64_unary_negate_no_rounding_cast() -> None:
    # the arith dtype must not route uint64 through float64
    vec = vectorize(make_fn("    a = max(x, True)\n    return (a + 0) * 1"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64))
    assert int(got[0]) == 9223372036854775809


# 3: unary negation covers maybe-boolean results


def test_unary_negate_maybe_bool() -> None:
    vec = vectorize(make_fn("    a = min(x, True)\n    return -a"))
    assert list(np.asarray(vec(np.asarray([True, False])))) == [-1, 0]


def test_unary_negate_maybe_bool_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = min(x, True)\n    return -a"))
    got = vec(xps.asarray([True, False]))
    assert list(map(int, np.asarray(got))) == [-1, 0]
