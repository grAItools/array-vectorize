"""Regression tests for the adversarial-review findings (round 13)."""

from __future__ import annotations

import numpy as np
from support import make_fn

from vectorizer import vectorize

# runtime-boolean min/max results participate in arithmetic conversion


def test_minmax_runtime_bool_arithmetic_numpy() -> None:
    # x is a boolean array at runtime, but a parameter statically: the
    # min/max result's bool dtype is only known at runtime
    vec = vectorize(make_fn("    a = min(x, True)\n    return a + a"))
    assert list(np.asarray(vec(np.asarray([True, False])))) == [2, 0]


def test_minmax_runtime_bool_arithmetic_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = min(x, True)\n    return a + a"))
    got = vec(xps.asarray([True, False]))
    assert list(map(int, np.asarray(got))) == [2, 0]


def test_minmax_runtime_bool_ref_chain() -> None:
    vec = vectorize(make_fn("    a = min(x, True)\n    b = a\n    return b + b"))
    assert list(np.asarray(vec(np.asarray([True, False])))) == [2, 0]


def test_minmax_runtime_numeric_noop() -> None:
    # the runtime intify is a no-op for numeric dtypes: min picks x (< 1)
    vec = vectorize(make_fn("    a = min(x, True)\n    return a + a"))
    assert np.allclose(vec(np.asarray([0.5, 0.0])), [1.0, 0.0])
    got = vec(np.asarray([0, -2], dtype=np.int32))
    assert list(np.asarray(got)) == [0, -4]


def test_minmax_runtime_bool_loop_carried() -> None:
    vec = vectorize(
        make_fn("    a = x\n    for i in range(1):\n        a = min(x, True)\n    return a + a")
    )
    assert list(np.asarray(vec(np.asarray([True, False])))) == [2, 0]


def test_minmax_runtime_bool_direct_call() -> None:
    vec = vectorize(make_fn("    return min(x, True) + min(x, True)"))
    assert list(np.asarray(vec(np.asarray([True, False])))) == [2, 0]
