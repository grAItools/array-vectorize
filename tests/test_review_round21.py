"""Regression tests for the adversarial-review findings (round 21)."""

from __future__ import annotations

import importlib.util
import itertools
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from vectorizer import vectorize

_tmp = tempfile.TemporaryDirectory(prefix="vec_rev21_")
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


# signed array operands with uint64: layered exactness


def test_uint64_plus_nonnegative_int64_array_exact() -> None:
    # huge uint64 + non-negative signed values: modular uint64
    vec = vectorize(make_fn("    a = max(x, True)\n    return a + y", defaults="x, y"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64), np.asarray([0], dtype=np.int64))
    assert int(got[0]) == 9223372036854775809


def test_uint64_plus_nonnegative_int64_array_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = max(x, True)\n    return a + y", defaults="x, y"))
    got = vec(xps.asarray([2**63 + 1], dtype=xps.uint64), xps.asarray([0], dtype=xps.int64))
    assert int(got[0]) == 9223372036854775809


def test_uint64_fit_plus_negative_int64_array_exact() -> None:
    # uint64 values that fit int64: int64 arithmetic handles negatives
    vec = vectorize(make_fn("    a = max(x, True)\n    return a + y", defaults="x, y"))
    got = vec(np.asarray([5], dtype=np.uint64), np.asarray([-3], dtype=np.int64))
    assert int(got[0]) == 2


def test_uint64_floordiv_signed_array_exact() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return a // y", defaults="x, y"))
    got = vec(np.asarray([7], dtype=np.uint64), np.asarray([-2], dtype=np.int64))
    assert int(got[0]) == -4


def test_uint64_mixed_sign_fallback_documented() -> None:
    # huge uint64 + negative signed: per-lane results fit no single
    # dtype; float64 approximation (documented divergence)
    vec = vectorize(make_fn("    a = max(x, True)\n    return a + y", defaults="x, y"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64), np.asarray([-1], dtype=np.int64))
    assert np.isclose(float(got[0]), 2**63)
