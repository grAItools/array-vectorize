"""Regression tests for the adversarial-review findings (round 22)."""

from __future__ import annotations

import importlib.util
import itertools
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from vectorizer import vectorize

_tmp = tempfile.TemporaryDirectory(prefix="vec_rev22_")
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


# positive signed-array divisors with huge uint64 dividends


def test_uint64_mod_positive_int64_array_exact() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return a % y", defaults="x, y"))
    got = vec(np.asarray([2**63 + 3], dtype=np.uint64), np.asarray([2], dtype=np.int64))
    assert int(got[0]) == 1


def test_uint64_floordiv_positive_int64_array_exact() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return a // y", defaults="x, y"))
    got = vec(np.asarray([2**63 + 3], dtype=np.uint64), np.asarray([2], dtype=np.int64))
    assert int(got[0]) == 4611686018427387905


def test_uint64_mod_positive_int64_array_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = max(x, True)\n    return a % y", defaults="x, y"))
    got = vec(xps.asarray([2**63 + 3], dtype=xps.uint64), xps.asarray([2], dtype=xps.int64))
    assert int(got[0]) == 1


def test_uint64_floordiv_positive_int64_array_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = max(x, True)\n    return a // y", defaults="x, y"))
    got = vec(xps.asarray([2**63 + 3], dtype=xps.uint64), xps.asarray([2], dtype=xps.int64))
    assert int(got[0]) == 4611686018427387905


def test_nonnegative_signed_left_mod_uint64_exact() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return y % a", defaults="x, y"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64), np.asarray([7], dtype=np.int64))
    assert int(got[0]) == 7
