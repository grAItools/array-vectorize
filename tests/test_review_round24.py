"""Regression tests for the adversarial-review findings (round 24)."""

from __future__ import annotations

import importlib.util
import itertools
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from vectorizer import vectorize

_tmp = tempfile.TemporaryDirectory(prefix="vec_rev24_")
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


# result-aware subtraction in the literal/unsigned-array path


def test_literal_minus_uint64_exact() -> None:
    vec = vectorize(make_fn("    return 0 - max(x, True)"))
    got = vec(np.asarray([1], dtype=np.uint64))
    assert int(got[0]) == -1


def test_literal_minus_uint64_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return 0 - max(x, True)"))
    got = vec(xps.asarray([1], dtype=xps.uint64))
    assert int(got[0]) == -1


def test_uint64_minus_uint64_exact() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return a - y", defaults="x, y"))
    got = vec(np.asarray([5], dtype=np.uint64), np.asarray([7], dtype=np.uint64))
    assert int(got[0]) == -2


def test_uint64_minus_uint64_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = max(x, True)\n    return a - y", defaults="x, y"))
    got = vec(xps.asarray([5], dtype=xps.uint64), xps.asarray([7], dtype=xps.uint64))
    assert int(got[0]) == -2


def test_sub_nonnegative_result_keeps_uint64() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return a - y", defaults="x, y"))
    got = vec(np.asarray([2**63 + 5], dtype=np.uint64), np.asarray([3], dtype=np.uint64))
    assert int(got[0]) == 2**63 + 2


def test_sub_mixed_magnitude_fallback_documented() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return a - y", defaults="x, y"))
    got = vec(np.asarray([2**63 + 5, 1], dtype=np.uint64), np.asarray([3, 4], dtype=np.uint64))
    assert np.isclose(float(got[0]), 2**63 + 2)
    assert np.isclose(float(got[1]), -3)
