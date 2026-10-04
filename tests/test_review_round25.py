"""Regression tests for the adversarial-review findings (round 25)."""

from __future__ import annotations

import numpy as np
from support import make_fn, make_module

from vectorizer import vectorize

# negative literals keep their sign in the result-range analysis


def test_uint64_minus_negative_literal_exact() -> None:
    vec = vectorize(make_fn("    return max(x, True) - -2"))
    got = vec(np.asarray([2**63 + 3], dtype=np.uint64))
    assert int(got[0]) == 2**63 + 5


def test_uint64_minus_negative_literal_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return max(x, True) - -2"))
    got = vec(xps.asarray([2**63 + 3], dtype=xps.uint64))
    assert int(got[0]) == 2**63 + 5


def test_negative_literal_minus_uint64_exact() -> None:
    vec = vectorize(make_fn("    return -2 - max(x, True)"))
    got = vec(np.asarray([5], dtype=np.uint64))
    assert int(got[0]) == -7


def test_negative_literal_minus_uint64_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return -2 - max(x, True)"))
    got = vec(xps.asarray([5], dtype=xps.uint64))
    assert int(got[0]) == -7


def test_negative_literal_sub_int64_min_exact() -> None:
    # -(2**63 - 1) - True == -(2**63): exactly int64-min
    vec = vectorize(
        make_module(
            "import math\n\n"
            "BOUND = -(2**63 - 1)\n\n"
            "def subject(x, y=2.0):\n"
            "    return BOUND - max(x, True)\n"
        ).subject
    )
    got = vec(np.asarray([0], dtype=np.uint64))
    assert int(got[0]) == -(2**63)


def test_negative_literal_sub_past_int64_min_fallback() -> None:
    # -(2**63) - True == -(2**63) - 1: fits no dtype, documented float64
    vec = vectorize(
        make_module(
            "import math\n\n"
            "BOUND = -(2**63)\n\n"
            "def subject(x, y=2.0):\n"
            "    return BOUND - max(x, True)\n"
        ).subject
    )
    got = vec(np.asarray([0], dtype=np.uint64))
    assert np.isclose(float(got[0]), -(2**63) - 1)
