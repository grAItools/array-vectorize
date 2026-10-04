"""Regression tests for the adversarial-review findings (round 14)."""

from __future__ import annotations

import importlib.util
import itertools
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from vectorizer import vectorize

_tmp = tempfile.TemporaryDirectory(prefix="vec_rev14_")
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


# 1: branch merges keep maybe-boolean tracking


def test_branch_merged_maybe_bool_arithmetic() -> None:
    vec = vectorize(
        make_fn(
            "    if x == True:\n"
            "        a = min(x, True)\n"
            "    else:\n"
            "        a = max(x, False)\n"
            "    return a + a"
        )
    )
    assert list(np.asarray(vec(np.asarray([True, False])))) == [2, 0]


# 2: maybe-boolean conversion preserves uint64 values


def test_maybe_bool_uint64_arithmetic_exact() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return a + 0"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64))
    assert int(got[0]) == 9223372036854775809


def test_maybe_bool_uint64_unary_negate_no_rounding_cast() -> None:
    # the arith dtype must not route uint64 through float64
    vec = vectorize(make_fn("    a = max(x, True)\n    return (a + 0) * 1"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64))
    assert int(got[0]) == 9223372036854775809


# 3: unary negation covers maybe-boolean results


def test_unary_negate_maybe_bool() -> None:
    vec = vectorize(make_fn("    a = min(x, True)\n    return -a"))
    assert list(np.asarray(vec(np.asarray([True, False])))) == [-1, 0]


def test_unary_negate_maybe_bool_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = min(x, True)\n    return -a"))
    got = vec(xps.asarray([True, False]))
    assert list(map(int, np.asarray(got))) == [-1, 0]
