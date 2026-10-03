"""Regression tests for the adversarial-review findings (round 7)."""

from __future__ import annotations

import importlib.util
import itertools
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from vectorizer import vectorize

_tmp = tempfile.TemporaryDirectory(prefix="vec_rev7_")
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


# 1: the identical-dtype fast path compares actual dtypes, not classes


def test_max_uint8_int16_no_narrowing() -> None:
    # uint8 and int16 share the promotion class (int, 16): the fast path
    # must not pick uint8 and wrap 300 to 44
    vec = vectorize(make_fn("    return max(x, y)"))
    got = vec(np.asarray([1], dtype=np.uint8), np.asarray([300], dtype=np.int16))
    assert got[0] == 300


def test_min_uint8_int16_negative() -> None:
    vec = vectorize(make_fn("    return min(x, y)"))
    got = vec(np.asarray([1], dtype=np.uint8), np.asarray([-1], dtype=np.int16))
    assert got[0] == -1


def test_max_bool_array_vs_literal() -> None:
    # bool and the literal 2 share the promotion class (int, 8): the fast
    # path must not pick bool and collapse 2 to True
    vec = vectorize(make_fn("    return max(x, 2)"))
    got = vec(np.asarray([False]))
    assert got[0] == 2


def test_min_uint64_vs_float64() -> None:
    # uint64 and float64 share the promotion class (float, 64): the fast
    # path must not pick uint64 and truncate 1.5 to 1
    vec = vectorize(make_fn("    return min(x, y)"))
    got = vec(np.asarray([2**63], dtype=np.uint64), np.asarray([1.5]))
    assert got[0] == 1.5


# 2: uint64 with a fitting literal stays exact


def test_max_uint64_literal_exact() -> None:
    # the literal 1 fits uint64, so the fast path keeps uint64: the
    # result is exact, not the float64 rounding of 9223372036854775809
    vec = vectorize(make_fn("    return max(x, 1)"))
    got = vec(np.asarray([9223372036854775809], dtype=np.uint64))
    assert got[0] == 9223372036854775809


# follow-ups: literal-fit precision at other widths


def test_min_float32_literal_needs_wider_dtype() -> None:
    # 16777217 is not exactly representable in float32: the fast path
    # must reject it and promote to float64
    vec = vectorize(make_fn("    return min(x, 16777217)"))
    got = vec(np.asarray([3e7], dtype=np.float32))
    assert got[0] == 16777217


def test_min_float32_exact_literal_keeps_float32() -> None:
    vec = vectorize(make_fn("    return min(x, 1.5)"))
    got = vec(np.asarray([2.0], dtype=np.float32))
    assert got[0] == 1.5


def test_min_int_literal_fit_boundaries() -> None:
    vec = vectorize(make_fn("    return min(x, 127)"))
    assert vec(np.asarray([100], dtype=np.int8))[0] == 100
    vec = vectorize(make_fn("    return max(x, 128)"))
    # 128 does not fit int8: promotion widens instead of wrapping
    assert vec(np.asarray([1], dtype=np.int8))[0] == 128
