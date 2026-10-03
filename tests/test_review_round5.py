"""Regression tests for the adversarial-review findings (round 5)."""

from __future__ import annotations

import importlib.util
import itertools
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from vectorizer import vectorize

_tmp = tempfile.TemporaryDirectory(prefix="vec_rev5_")
_TMPDIR = Path(_tmp.name)
_seq = itertools.count()


def make_fn(body: str, defaults: str = "x, y=2.0") -> Callable[..., Any]:
    path = _TMPDIR / f"snippet_{next(_seq)}.py"
    path.write_text("import math\n\n\ndef subject(" + defaults + "):\n" + body + "\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.subject


# 1: min/max literal promotion never truncates or overflows


def test_min_float_literal_int_input() -> None:
    # the float bound must widen the common dtype, not truncate to int64
    vec = vectorize(make_fn("    return min(x, 1.5)"))
    got = vec(np.asarray([2], dtype=np.int64))
    assert got[0] == 1.5


def test_max_literal_int8_no_overflow() -> None:
    # 300 must widen int8, not wrap to 44
    vec = vectorize(make_fn("    return max(x, 300)"))
    got = vec(np.asarray([1], dtype=np.int8))
    assert got[0] == 300


# 2: compound min/max siblings


def test_min_compound_sibling_exact_big_int() -> None:
    vec = vectorize(make_fn("    return min(x + 0, 9007199254740993)"))
    got = vec(np.asarray([9007199254740994], dtype=np.int64))
    assert got[0] == 9007199254740993


# 3 + 4: mixed loop kinds keep float values through assignments and division


def test_float_mixed_propagates_through_assignment() -> None:
    vec = vectorize(
        make_fn(
            "    b = x\n    for i in range(1):\n        b = x + 0.5\n    c = b\n    return c + c"
        )
    )
    assert np.allclose(vec(np.asarray([1.0, 2.0])), [3.0, 5.0])


def test_true_division_loop_mix_keeps_float() -> None:
    vec = vectorize(
        make_fn(
            "    b = x > 0\n"
            "    a = x\n"
            "    for i in range(2):\n"
            "        a = b + b\n"
            "        b = x / 2.0\n"
            "    return a"
        )
    )
    assert vec(np.asarray([3.0]))[0] == 3.0


# 5: boolean bitwise expressions stay boolean


def test_bool_bitwise_stays_bool() -> None:
    vec = vectorize(make_fn("    b = (x > 0) & (x < 3)\n    return b + b"))
    got = vec(np.asarray([1.0]))
    assert got[0] == 2


def test_bool_bitwise_or_xor_stay_bool() -> None:
    vec = vectorize(make_fn("    return ((x > 0) | (x > 1)) + ((x > 0) ^ (x > 1))"))
    got = vec(np.asarray([1.0]))
    assert got[0] == 2


# 6: bool + float-array arithmetic on strict backends


def test_bool_plus_float_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return (x > 0) + x"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 2.0


def test_bool_plus_float_literal_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return (x > 0) + 1.5"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 2.5


# 7: scalar tracking through loop phis


def test_scalar_phi_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(
        make_fn(
            "    b = 4.0\n"
            "    for i in range(0):\n"
            "        if x > 0:\n"
            "            b = 9.0\n"
            "    return math.sqrt(b)"
        )
    )
    got = vec(xps.asarray([1.0]))
    # constant w.r.t. x: 0-d result is broadcast for comparison
    assert np.allclose(np.broadcast_to(np.asarray(got, dtype=float), (1,)), [2.0])


# round-5 follow-ups: min/max promotion matrix stays exact


def test_minmax_promotion_matrix_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return min(x, 3)"))
    assert int(vec(xps.asarray([1], dtype=xps.int64))[0]) == 1

    vec = vectorize(make_fn("    return min(x, 1.5) + max(y, 1)"))
    got = vec(xps.asarray([2.0]))
    assert float(got[0]) == 3.5  # min(2, 1.5) + max(2, 1)

    vec = vectorize(make_fn("    return min(x, y)"))
    got = vec(xps.asarray([1.0, 3.0]), xps.asarray([2.0, 2.0]))
    assert np.allclose(np.asarray(got, dtype=float), [1.0, 2.0])
