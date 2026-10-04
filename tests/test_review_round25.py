"""Regression tests for the adversarial-review findings (round 25)."""

from __future__ import annotations

import importlib.util
import itertools
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from vectorizer import vectorize

_tmp = tempfile.TemporaryDirectory(prefix="vec_rev25_")
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


def make_module(source: str) -> Callable[..., Any]:
    path = _TMPDIR / f"module_{next(_seq)}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.subject


# negative literals keep their sign in the result-range analysis


def test_uint64_minus_negative_literal_exact() -> None:
    vec = vectorize(make_fn("    return max(x, True) - -2"))
    got = vec(np.asarray([2**63 + 3], dtype=np.uint64))
    assert int(got[0]) == 2**63 + 5


def test_uint64_minus_negative_literal_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return max(x, True) - -2"))
    got = vec(xps.asarray([2**63 + 3], dtype=xps.uint64))
    assert int(got[0]) == 2**63 + 5


def test_negative_literal_minus_uint64_exact() -> None:
    vec = vectorize(make_fn("    return -2 - max(x, True)"))
    got = vec(np.asarray([5], dtype=np.uint64))
    assert int(got[0]) == -7


def test_negative_literal_minus_uint64_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return -2 - max(x, True)"))
    got = vec(xps.asarray([5], dtype=xps.uint64))
    assert int(got[0]) == -7


def test_negative_literal_sub_int64_min_exact() -> None:
    # -(2**63 - 1) - True == -(2**63): exactly int64-min
    vec = vectorize(
        make_module(
            "import math\n\n"
            "BOUND = -(2**63 - 1)\n\n"
            "def subject(x, y=2.0):\n"
            "    return BOUND - max(x, True)\n"
        )
    )
    got = vec(np.asarray([0], dtype=np.uint64))
    assert int(got[0]) == -(2**63)


def test_negative_literal_sub_past_int64_min_fallback() -> None:
    # -(2**63) - True == -(2**63) - 1: fits no dtype, documented float64
    vec = vectorize(
        make_module(
            "import math\n\n"
            "BOUND = -(2**63)\n\n"
            "def subject(x, y=2.0):\n"
            "    return BOUND - max(x, True)\n"
        )
    )
    got = vec(np.asarray([0], dtype=np.uint64))
    assert np.isclose(float(got[0]), -(2**63) - 1)
