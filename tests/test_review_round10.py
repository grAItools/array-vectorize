"""Regression tests for the adversarial-review findings (round 10)."""

from __future__ import annotations

import importlib.util
import itertools
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from vectorizer import vectorize

_tmp = tempfile.TemporaryDirectory(prefix="vec_rev10_")
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


def py_scalar(got: Any) -> int:
    return int(np.asarray(got).reshape(-1)[0])


# 1: integer-only min/max never loses precision through uint64 promotion


def test_min_uint64_always_winning_literal_exact() -> None:
    # the negative bound always wins: the result is the literal, exactly
    # representable in int64 (float64 would round -9007199254740993)
    vec = vectorize(make_fn("    return min(x, -9007199254740993)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64))
    assert py_scalar(got) == -9007199254740993


def test_min_uint64_result_supports_bitwise() -> None:
    # the min result must stay INTEGER so downstream int ops work
    vec = vectorize(make_fn("    return min(x, -1) & 1"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64))
    assert py_scalar(got) == 1


def test_max_uint64_int64_arrays_exact() -> None:
    vec = vectorize(make_fn("    return max(x, y)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64), np.asarray([-1], dtype=np.int64))
    assert py_scalar(got) == 2**63 + 1


def test_min_uint64_int64_arrays_exact() -> None:
    vec = vectorize(make_fn("    return min(x, y)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64), np.asarray([-1], dtype=np.int64))
    assert py_scalar(got) == -1


# follow-ups: the integer result-range invariants


def test_min_uint64_signed_positive_literals() -> None:
    # min results fit int64 when something signed can win
    vec = vectorize(make_fn("    return min(x, -1, 5)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64))
    assert py_scalar(got) == -1


def test_max_uint64_all_unsigned_widens() -> None:
    vec = vectorize(make_fn("    return max(x, y)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64), np.asarray([7], dtype=np.uint8))
    assert py_scalar(got) == 2**63 + 1


def test_min_max_roundtrip_prior_findings() -> None:
    # the whole prior promotion matrix stays exact after the restructure
    vec = vectorize(make_fn("    return max(x, -1)"))
    assert py_scalar(vec(np.asarray([2**63 + 1], dtype=np.uint64))) == 9223372036854775809
    vec = vectorize(make_fn("    return max(x, 1)"))
    assert py_scalar(vec(np.asarray([9223372036854775809], dtype=np.uint64))) == 9223372036854775809
    vec = vectorize(make_fn("    return min(x, 300)"))
    assert py_scalar(vec(np.asarray([1], dtype=np.uint8))) == 1
    vec = vectorize(make_fn("    return max(x, y)"))
    assert py_scalar(vec(np.asarray([1], dtype=np.uint8), np.asarray([300], dtype=np.int16))) == 300
    vec = vectorize(make_fn("    return min(x, 1.5)"))
    assert vec(np.asarray([2], dtype=np.int64))[0] == 1.5
    vec = vectorize(make_fn("    return max(x, 2049.0)"))
    assert vec(np.asarray([0.0], dtype=np.float16))[0] == 2049.0


# helper-branch coverage: fit-and-clamp literals, u64-max literals, f16 overflow


def test_fit_literal_alongside_clamping_literal() -> None:
    # 5 fits uint8 while -1 clamps: both stay in the shared dtype
    vec = vectorize(make_fn("    return max(x, 5, -1)"))
    got = vec(np.asarray([1], dtype=np.uint8))
    assert py_scalar(got) == 5


def test_max_uint64_mixed_arrays_negative_literal() -> None:
    vec = vectorize(make_fn("    return max(x, y, -1)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64), np.asarray([-5], dtype=np.int64))
    assert py_scalar(got) == 2**63 + 1


def test_max_uint64_mixed_arrays_positive_literal() -> None:
    vec = vectorize(make_fn("    return max(x, y, 3)"))
    got = vec(np.asarray([1], dtype=np.uint64), np.asarray([-5], dtype=np.int64))
    assert py_scalar(got) == 3


def test_min_float16_overflow_literal_promotes() -> None:
    # 1e10 overflows float16 in the IEEE round trip: promote to float64
    vec = vectorize(make_fn("    return min(x, 1e10)"))
    got = vec(np.asarray([0.0], dtype=np.float16))
    assert got[0] == 0.0


def test_fits_dtype_defensive_branches() -> None:
    from vectorizer._runtime import _fits_dtype

    class _FakeDtype:
        def __init__(self, name: str) -> None:
            self._name = name

        def __str__(self) -> str:
            return self._name

    assert not _fits_dtype(1.5, _FakeDtype("float24"))  # exotic width: promote
    assert not _fits_dtype("not-a-number", np.dtype("int64"))
    assert _fits_dtype(True, np.dtype("bool"))
    assert not _fits_dtype(1, np.dtype("bool"))
