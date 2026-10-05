"""Math and builtin calls, casts, pow, floor-mod, abs/round, scalar-argument math."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
from support import make_fn

import vectorizer
from vectorizer import vectorize

spec = importlib.util.spec_from_file_location(
    "vec_corpus_b", Path(__file__).parent.parent / "corpus.py"
)
assert spec is not None and spec.loader is not None
CORPUS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CORPUS)


def vfn(name: str) -> Any:
    return vectorize(getattr(CORPUS, name))


X = np.asarray([-2.0, -0.5, 0.0, 0.5, 2.0, 10.0])
Y = np.asarray([3.0, 1.5, 1.0, 0.5, -1.0, -3.0])


# ---- from test_behavior (git history: tests/test_behavior.py)


def test_floor_mod_ops() -> None:
    expected = np.floor_divide(X, 2) + np.remainder(X, 3)
    got = vfn("floor_mod_ops")(X)
    assert np.allclose(got, expected)


def test_pow_op() -> None:
    expected = np.power(X, 2) + np.power(2.0, X)
    assert np.allclose(vfn("pow_op")(X), expected)


def test_math_calls() -> None:
    mx = X * (X > 0)  # keep sqrt domain valid
    expected = np.sqrt(mx) + np.exp(mx) + np.log1p(mx) + np.sin(mx)
    got = vfn("math_calls")(mx)
    with np.errstate(invalid="ignore"):
        assert np.allclose(got, expected, equal_nan=True)


def test_math_two_arg() -> None:
    expected = np.arctan2(X, Y) + np.hypot(X, Y) + np.copysign(X, Y)
    assert np.allclose(vfn("math_two_arg")(X, Y), expected)


def test_casts() -> None:
    got = vfn("casts")(X)
    expected = X.astype(np.int64) + X + X.astype(bool) + X.astype(np.int64)
    assert np.allclose(got, expected)


def test_abs_round() -> None:
    assert np.allclose(vfn("abs_round")(X), np.abs(X) + np.round(X))


def test_complex_expr() -> None:
    t = np.exp(-(X * X + Y * Y))
    expected = t / (1.0 + t)
    assert np.allclose(vfn("complex_expr")(X, Y), expected)


# ---- from test_m4 (git history: tests/test_m4.py)


def test_div_mod_sign_semantics_match_python() -> None:
    fn_mod = make_fn("    return x % 3")
    fn_floordiv = make_fn("    return x // 3")
    xs = np.asarray([-7.0, -6.5, 6.5, 7.0])
    assert np.allclose(fn_mod and vectorize(fn_mod)(xs), [x % 3 for x in xs])
    assert np.allclose(vectorize(fn_floordiv)(xs), [x // 3 for x in xs])


def test_round_half_even() -> None:
    fn = make_fn("    return round(x)")
    xs = np.asarray([0.5, 1.5, 2.5, -0.5, -1.5])
    assert np.array_equal(vectorize(fn)(xs), [round(x) for x in xs])  # banker's rounding


# ---- from test_smoke (git history: tests/test_smoke.py)


def test_import() -> None:
    assert vectorizer.__version__


# ---- from test_review_round3 (git history: tests/test_review_round3.py)


def test_cast_scalar_default_param() -> None:
    vec = vectorize(make_fn("    return x + int(y)"))
    got = vec(np.asarray([1.0]))
    assert got[0] == 3.0  # y defaults to 2.0 -> int(2.0) == 2


def test_cast_literal_folds() -> None:
    vec = vectorize(make_fn("    return x + int(3.0)"))
    got = vec(np.asarray([1.0]))
    assert got[0] == 4.0


def test_trunc_scalar_default_param() -> None:
    vec = vectorize(make_fn("    return x + math.trunc(y)"))
    got = vec(np.asarray([1.0]))
    assert got[0] == 3.0  # math.trunc(2.0) == 2


def test_abs_round_loop_var() -> None:
    vec = vectorize(
        make_fn(
            "    s = x\n"
            "    for i in range(3):\n"
            "        s = s + abs(i - 1) + round(i / 2)\n"
            "    return s"
        )
    )
    want = 0.0
    for i in range(3):
        want += abs(i - 1) + round(i / 2)
    assert vec(np.asarray([0.0]))[0] == want
