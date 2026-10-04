"""Regression tests for the adversarial-review findings (round 19)."""

from __future__ import annotations

import numpy as np
from support import make_fn

from vectorizer import vectorize

# 1: correct remainder formula for negative divisors


def test_uint64_mod_negative_divisor_formula() -> None:
    # 5 % -3 == -1 (not -(5 % 3) == -2); 6 % -3 == 0
    vec = vectorize(make_fn("    return max(x, True) % -3"))
    assert int(vec(np.asarray([5], dtype=np.uint64))[0]) == -1
    assert int(vec(np.asarray([6], dtype=np.uint64))[0]) == 0


# 2: floor-division corrections use numeric casts (strict-safe)


def test_uint64_floordiv_negative_divisor_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return max(x, True) // -3"))
    got = vec(xps.asarray([5], dtype=xps.uint64))
    assert int(got[0]) == -2


def test_uint64_negative_literal_floordiv_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return -3 // max(x, True)"))
    got = vec(xps.asarray([5], dtype=xps.uint64))
    assert int(got[0]) == -1


# 3: literal-left remainder keeps exactly representable uint64 results


def test_negative_literal_mod_uint64_exact() -> None:
    # -1 % (2**63 + 2) == 2**63 + 1: fits uint64 (not int64)
    vec = vectorize(make_fn("    return -1 % max(x, True)"))
    got = vec(np.asarray([2**63 + 2], dtype=np.uint64))
    assert int(got[0]) == 9223372036854775809


def test_negative_literal_mod_small_array() -> None:
    vec = vectorize(make_fn("    return -2 % max(x, True)"))
    got = vec(np.asarray([2**53 + 1, 2**63 + 1], dtype=np.uint64))
    assert list(map(int, np.asarray(got))) == [2**53 - 1, 2**63 - 1]
