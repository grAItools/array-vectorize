"""Regression tests for the adversarial-review findings (round 16)."""

from __future__ import annotations

import numpy as np
from support import make_fn

from vectorizer import vectorize

# the uint64 modular cast respects operator semantics


def test_uint64_mod_negative_divisor() -> None:
    # -2 % 3 == 1: a negative divisor changes mod semantics, so the
    # uint64 modular path must not wrap the literal
    vec = vectorize(make_fn("    a = max(x, True)\n    return -2 % a"))
    got = vec(np.asarray([3], dtype=np.uint64))
    assert got[0] == 1


def test_uint64_floordiv_negative_divisor() -> None:
    # 3 // -2 == -2: floor division with a negative divisor
    vec = vectorize(make_fn("    a = max(x, True)\n    return a // -2"))
    got = vec(np.asarray([3], dtype=np.uint64))
    assert got[0] == -2.0


def test_uint64_truediv_negative_divisor() -> None:
    # 3 / -2 == -1.5: true division is float, and the operands must be
    # cast to float (strict rejects integer true division of u64 wraps)
    vec = vectorize(make_fn("    a = max(x, True)\n    return a / -2"))
    got = vec(np.asarray([3], dtype=np.uint64))
    assert got[0] == -1.5


def test_uint64_truediv_negative_divisor_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = max(x, True)\n    return a / -2"))
    got = vec(xps.asarray([3], dtype=xps.uint64))
    assert float(got[0]) == -1.5


def test_uint64_floordiv_mod_positive_keep_uint64() -> None:
    # non-negative divisors keep the exact modular uint64 path
    vec = vectorize(make_fn("    a = max(x, True)\n    return a // 2"))
    got = vec(np.asarray([2**63 + 3], dtype=np.uint64))
    assert int(got[0]) == (2**63 + 3) // 2
    vec = vectorize(make_fn("    a = max(x, True)\n    return a % 2"))
    got = vec(np.asarray([2**63 + 3], dtype=np.uint64))
    assert int(got[0]) == 1


def test_uint64_add_negative_literal_still_modular() -> None:
    # add/sub/mul keep the modular path for negative literals
    vec = vectorize(make_fn("    a = max(x, True)\n    return a + -2"))
    got = vec(np.asarray([2**63 + 3], dtype=np.uint64))
    assert int(got[0]) == 9223372036854775809
