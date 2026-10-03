"""Regression tests for the adversarial-review findings (round 6)."""

from __future__ import annotations

import importlib.util
import itertools
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from vectorizer import vectorize

_tmp = tempfile.TemporaryDirectory(prefix="vec_rev6_")
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


# 1: intify preserves the converted operand's own dtype


def test_mixed_loop_operand_not_truncated_by_literal() -> None:
    vec = vectorize(
        make_fn("    b = x\n    for i in range(1):\n        b = x + 0.5\n    return b + 1")
    )
    assert np.allclose(vec(np.asarray([1.0, 2.0])), [2.5, 3.5])


def test_mixed_loop_operand_unary_negate() -> None:
    vec = vectorize(
        make_fn("    b = x\n    for i in range(1):\n        b = x + 0.5\n    return -b")
    )
    assert np.allclose(vec(np.asarray([1.0, 2.0])), [-1.5, -2.5])


# 2: bool arithmetic gets int64 headroom, not int8


def test_bool_arith_times_hundred() -> None:
    vec = vectorize(make_fn("    return ((x > 0) + (x > 0)) * 100"))
    assert vec(np.asarray([1.0]))[0] == 200


def test_bool_loop_count_128() -> None:
    vec = vectorize(
        make_fn("    s = 0\n    for i in range(128):\n        s = s + (x > 0)\n    return s")
    )
    assert vec(np.asarray([1.0]))[0] == 128


# 3: uint64 promotion never produces negatives


def test_min_uint64_vs_literal() -> None:
    vec = vectorize(make_fn("    return min(x, 1)"))
    got = vec(np.asarray([2**63], dtype=np.uint64))
    assert got[0] == 1


# 4: int64 precision beats float32 rank in min/max promotion


def test_min_float32_vs_int64_exact() -> None:
    vec = vectorize(make_fn("    return min(x, y)"))
    got = vec(np.asarray([20000000], dtype=np.float32), np.asarray([16777217], dtype=np.int64))
    assert got[0] == 16777217


# 5: helper names cannot be shadowed by user parameters


def test_helper_name_shadowed_by_param() -> None:
    vec = vectorize(make_fn("    return min(_vec_common_dtype, 3)", defaults="_vec_common_dtype"))
    got = vec(np.asarray([1.0]))
    assert got[0] == 1.0


def test_helper_names_shadowed_by_params() -> None:
    vec = vectorize(
        make_fn(
            "    return min(_vec_arith_dtype, _vec_common_dtype) + (_vec_arith_dtype > 0)",
            defaults="_vec_arith_dtype, _vec_common_dtype=2.0",
        )
    )
    got = vec(np.asarray([1.0]))
    assert got[0] == 2.0  # min(1, 2) + True


# 6: helper cache respects protect_domains


def test_helper_cache_protect_flag() -> None:
    path = _TMPDIR / "cache_protect.py"
    path.write_text(
        "import math\n\n\n"
        "def inner(x):\n"
        "    return math.sqrt(x) if x > 0 else 0.0\n\n\n"
        "def outer(x):\n"
        "    return inner(x)\n"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vectorize(mod.outer)  # caches the unprotected inner
    protected = vectorize(mod.outer, protect_domains=True)
    with np.errstate(all="raise"):
        got = protected(np.asarray([-1.0, 4.0]))
    assert np.allclose(got, [0.0, 2.0])


# round-6 follow-ups: promotion rules stay exact


def test_minmax_identical_dtypes_passthrough() -> None:
    vec = vectorize(make_fn("    return min(x, y)"))
    got = vec(np.asarray([3, 1], dtype=np.int8), np.asarray([2, 2], dtype=np.int8))
    assert list(np.asarray(got)) == [2, 1]


def test_bool_plus_int8_array_headroom() -> None:
    vec = vectorize(make_fn("    return (x > 0) + y", defaults="x, y"))
    got = vec(np.asarray([1.0]), np.asarray([100], dtype=np.int8))
    assert got[0] == 101
