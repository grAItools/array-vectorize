"""Regression tests for the adversarial-review findings (round 20)."""

from __future__ import annotations

import numpy as np
from support import make_fn, make_module

from vectorizer import vectorize

# 1: remainder by int64-min (the negative divisor is added directly)


def test_uint64_mod_int64_min_divisor() -> None:
    mod = make_module(
        "import math\n\n"
        "BOUND = -(2**63)\n\n"
        "def subject(x, y=2.0):\n"
        "    return max(x, True) % BOUND\n"
    )
    vec = vectorize(mod.subject)
    got = vec(np.asarray([1], dtype=np.uint64))
    assert int(got[0]) == -9223372036854775807


def test_uint64_mod_int64_min_divisor_strict() -> None:
    import array_api_strict as xps

    mod = make_module(
        "import math\n\n"
        "BOUND = -(2**63)\n\n"
        "def subject(x, y=2.0):\n"
        "    return max(x, True) % BOUND\n"
    )
    vec = vectorize(mod.subject)
    got = vec(xps.asarray([1], dtype=xps.uint64))
    assert int(got[0]) == -9223372036854775807


# 2: arithmetic helper results keep kind information


def test_arith_result_negative_power() -> None:
    # (x > 0) + 1 is provably int: the negative-exponent float cast
    # must fire on the helper result
    vec = vectorize(make_fn("    a = (x > 0) + 1\n    return a ** -1"))
    assert vec(np.asarray([2]))[0] == 0.5


def test_arith_result_negative_power_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = (x > 0) + 1\n    return a ** -1"))
    assert float(vec(xps.asarray([2]))[0]) == 0.5


def test_arith_result_float_kind() -> None:
    # float operands keep float: no int-power issue to begin with
    vec = vectorize(make_fn("    a = (x > 0) + 0.5\n    return a ** -1"))
    assert vec(np.asarray([2]))[0] == 1 / 1.5  # True + 0.5 == 1.5
