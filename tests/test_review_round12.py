"""Regression tests for the adversarial-review findings (round 12)."""

from __future__ import annotations

import importlib.util
import itertools
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from vectorizer import vectorize

_tmp = tempfile.TemporaryDirectory(prefix="vec_rev12_")
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
