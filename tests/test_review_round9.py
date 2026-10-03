"""Regression tests for the adversarial-review findings (round 9)."""

from __future__ import annotations

import importlib.util
import itertools
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from vectorizer import vectorize

_tmp = tempfile.TemporaryDirectory(prefix="vec_rev9_")
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


# 1: negative literals participate in min/max literal handling


def test_max_uint64_negative_literal_exact() -> None:
    # `-1` lowers to UnaryOp(neg, 1): it must be recognized as a literal
    # so the never-winning-bound clamp fires (float64 would round
    # 9223372036854775809 to ...808)
    vec = vectorize(make_fn("    return max(x, -1)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64))
    # exact comparison via Python int (numpy scalar == rounds floats)
    assert int(got[0]) == 9223372036854775809


def test_min_uint64_negative_literal_wins_exactly() -> None:
    vec = vectorize(make_fn("    return min(x, -1)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64))
    assert int(got[0]) == -1


def test_negative_literal_other_contexts() -> None:
    vec = vectorize(make_fn("    return min(x, -1) + max(y, -2.5)"))
    got = vec(np.asarray([3.0]), np.asarray([-3.0]))
    assert got[0] == -3.5  # min(3, -1) + max(-3, -2.5) = -1 + -2.5


# 2: variadic min/max literals agree on the final common dtype


def test_max_variadic_literals_strict() -> None:
    import array_api_strict as xps

    # 1 fits uint8 but 1.5 forces float64: BOTH literals must cast to
    # float64 (a uint8 literal would break strict same-dtype promotion)
    vec = vectorize(make_fn("    return max(x, 1, 1.5)"))
    got = vec(xps.asarray([2], dtype=xps.uint8))
    assert int(got[0]) == 2


def test_min_variadic_negative_and_float_literals() -> None:
    vec = vectorize(make_fn("    return min(x, -1, 1.5)"))
    got = vec(np.asarray([2], dtype=np.uint8))
    assert int(got[0]) == -1


def test_max_variadic_fitting_literals_keep_dtype() -> None:
    vec = vectorize(make_fn("    return max(x, 1, 2)"))
    got = vec(np.asarray([0], dtype=np.uint8))
    assert int(got[0]) == 2
