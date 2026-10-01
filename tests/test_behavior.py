"""Behavioral tests on NumPy (plan T3): vectorized output vs hand-computed arrays."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vectorizer import vectorize

spec = importlib.util.spec_from_file_location("vec_corpus_b", Path(__file__).parent / "corpus.py")
assert spec is not None and spec.loader is not None
CORPUS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CORPUS)


def vfn(name: str) -> Any:
    return vectorize(getattr(CORPUS, name))


X = np.asarray([-2.0, -0.5, 0.0, 0.5, 2.0, 10.0])
Y = np.asarray([3.0, 1.5, 1.0, 0.5, -1.0, -3.0])


def test_add() -> None:
    assert np.allclose(vfn("add")(X, Y), X + Y)


def test_arith_ops() -> None:
    expected = (X + Y - X * Y) / (X * X + Y * Y)
    assert np.allclose(vfn("arith_ops")(X, Y), expected)


def test_floor_mod_ops() -> None:
    expected = np.floor_divide(X, 2) + np.remainder(X, 3)
    got = vfn("floor_mod_ops")(X)
    assert np.allclose(got, expected)


def test_pow_op() -> None:
    expected = np.power(X, 2) + np.power(2.0, X)
    assert np.allclose(vfn("pow_op")(X), expected)


def test_bitwise_ops() -> None:
    xi = np.asarray([-3, 2, 5, 7], dtype=np.int64)
    yi = np.asarray([4, 1, 6, 2], dtype=np.int64)
    expected = (xi & yi) | (xi ^ 3) << 1 >> 2
    assert np.array_equal(vfn("bitwise_ops")(xi, yi), expected)


def test_unary_ops_int() -> None:
    xi = np.asarray([-3, 2, 5], dtype=np.int64)
    assert np.array_equal(vfn("unary_ops")(xi), -xi + (+xi) - (~xi))


def test_math_calls() -> None:
    mx = X * (X > 0)  # keep sqrt domain valid
    expected = np.sqrt(mx) + np.exp(mx) + np.log1p(mx) + np.sin(mx)
    got = vfn("math_calls")(mx)
    with np.errstate(invalid="ignore"):
        assert np.allclose(got, expected, equal_nan=True)


def test_math_two_arg() -> None:
    expected = np.arctan2(X, Y) + np.hypot(X, Y) + np.copysign(X, Y)
    assert np.allclose(vfn("math_two_arg")(X, Y), expected)


def test_minmax() -> None:
    expected = np.minimum(X, Y) + np.maximum(np.maximum(X, Y), 3.0)
    assert np.allclose(vfn("minmax")(X, Y), expected)


def test_casts() -> None:
    got = vfn("casts")(X)
    expected = X.astype(np.int64) + X + X.astype(bool) + X.astype(np.int64)
    assert np.allclose(got, expected)


def test_abs_round() -> None:
    assert np.allclose(vfn("abs_round")(X), np.abs(X) + np.round(X))


def test_compare_chain() -> None:
    got = vfn("compare_chain")(X)
    assert got.dtype == np.bool_
    assert np.array_equal(got, (X > 0) & (X < 1))


def test_boolop_logical() -> None:
    got = vfn("boolop_logical")(X, Y)
    expected = ((X > 0) & (Y > 0)) | ~(X < 0)
    assert np.array_equal(got, expected)


def test_boolop_numeric_exact_including_nan() -> None:
    arr = np.asarray([-0.0, 0.0, 2.0, np.nan])
    got = vfn("boolop_numeric")(arr, np.asarray([5.0, 5.0, 0.0, 1.0]))
    # Python: x and y -> x if falsy else y; NaN is truthy
    expected = np.asarray([arr[0], 0.0, 0.0, 1.0])
    assert np.array_equal(got, expected, equal_nan=True)


def test_ternary() -> None:
    expected = np.where(X > 0, X * 2, X / 2)
    assert np.allclose(vfn("ternary")(X), expected)


def test_ssa_rebind() -> None:
    expected = (X + 1) * 2 + 3
    assert np.allclose(vfn("ssa_rebind")(X), expected)


def test_closure_scalar() -> None:
    assert np.allclose(vfn("closure_scalar")(X), X * CORPUS.SCALE)


def test_closure_array() -> None:
    got = vfn("closure_array")(np.asarray([1.0, 1.0, 1.0]))
    assert np.allclose(got, np.asarray([2.0, 3.0, 4.0]))


@pytest.mark.parametrize("name", ["relu", "relu_with_else"])
def test_relu_variants(name: str) -> None:
    assert np.allclose(vfn(name)(X), np.maximum(X, 0.0))


def test_psi() -> None:
    expected = np.where(X < 0, 0.0, X * np.exp(-X))
    assert np.allclose(vfn("psi")(X), expected)


def test_clamp() -> None:
    assert np.allclose(vfn("clamp")(X), np.clip(X, 0.0, 1.0))
    assert np.allclose(vfn("clamp")(X, -1.0, 0.5), np.clip(X, -1.0, 0.5))
    assert np.allclose(vfn("clamp")(X, hi=0.25), np.clip(X, 0.0, 0.25))


def test_piecewise() -> None:
    expected = np.where(X < -1, -1.0, np.where(X > 1, 1.0, X))
    assert np.allclose(vfn("piecewise")(X), expected)


def test_merge_with_prior() -> None:
    assert np.allclose(vfn("merge_with_prior")(X), np.where(X > 0, X, 0.0))


def test_nested_early_returns() -> None:
    expected = np.where(X > 0, np.where(X > 10, 1.0, 2.0), 3.0)
    assert np.allclose(vfn("nested_early_returns")(X), expected)


def test_both_branches_assign() -> None:
    expected = np.where(X > 0, X, -X) + np.where(X > 0, 1.0, 2.0)
    assert np.allclose(vfn("both_branches_assign")(X), expected)


def test_kwonly_defaults() -> None:
    expected = X * 2.0 + 0.5
    assert np.allclose(vfn("kwonly_defaults")(X), expected)
    expected = X * 4.0 + 1.0
    assert np.allclose(vfn("kwonly_defaults")(X, scale=4.0, bias=1.0), expected)


def test_reserved_param() -> None:
    assert np.allclose(vfn("reserved_param")(X), X + 1)


def test_cse_opportunity() -> None:
    s = np.sqrt(np.abs(X))
    expected = s * s + s
    assert np.allclose(vfn("cse_opportunity")(np.abs(X)), expected)


def test_complex_expr() -> None:
    t = np.exp(-(X * X + Y * Y))
    expected = t / (1.0 + t)
    assert np.allclose(vfn("complex_expr")(X, Y), expected)


def test_scalar_python_args_accepted() -> None:
    # Python scalars ride along via standard promotion
    assert np.allclose(vfn("add")(X, 2.0), X + 2.0)


def test_backend_array_api_strict() -> None:
    import array_api_strict as xps

    a = xps.asarray([1.0, -2.0, 3.0])
    got = vfn("relu")(a)
    assert isinstance(got, xps.asarray([1.0]).__class__)
    assert list(map(float, got)) == [1.0, 0.0, 3.0]


def test_divergence_div_by_zero_is_inf() -> None:
    with np.errstate(divide="ignore"):
        got = vectorize(lambda x: 1.0 / x)(np.asarray([-1.0, 0.0, 2.0]))
    assert np.allclose(got, [-1.0, np.inf, 0.5])


def test_divergence_neg_sqrt_is_nan() -> None:
    with np.errstate(invalid="ignore"):
        got = vfn("cse_opportunity")(np.asarray([-4.0]))
    assert np.isnan(got[0])
