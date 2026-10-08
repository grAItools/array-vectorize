# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Differential tests vs the scalar original using Hypothesis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import corpus
import hypothesis
from hypothesis import strategies as st
import numpy as np

import array_vectorize

CORPUS = corpus

# bounded floats keep scalar math from raising (exp overflow, log domain)
finite = st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False)
small = st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False)
positive = st.floats(min_value=0.0, max_value=50, allow_nan=False, allow_infinity=False)


def arrays(
    value_strategy: st.SearchStrategy[float], size: int = 12
) -> st.SearchStrategy[np.ndarray]:
    return st.lists(value_strategy, min_size=size, max_size=size).map(
        lambda xs: np.asarray(xs, dtype=np.float64)
    )


def check(fn: Any, xs: np.ndarray, *rest: np.ndarray) -> None:
    vec = array_vectorize.vectorize(fn)
    got = vec(xs, *rest)
    pairs: list[tuple[Any, ...]] = list(zip(xs, *rest, strict=True)) if rest else [(x,) for x in xs]
    expected = np.asarray([fn(*pair) for pair in pairs])
    assert np.allclose(got, expected, equal_nan=True, rtol=1e-12), (got, expected)


@hypothesis.given(arrays(finite))
def test_relu_diff(xs: np.ndarray) -> None:
    check(CORPUS.relu, xs)


@hypothesis.given(arrays(finite))
def test_psi_diff(xs: np.ndarray) -> None:
    check(CORPUS.psi, xs)


@hypothesis.given(arrays(small), arrays(small))
def test_clamp_diff(xs: np.ndarray, lo: np.ndarray) -> None:
    vec = array_vectorize.vectorize(CORPUS.clamp)
    lo_arr = np.minimum(np.maximum(lo, -2.0), 0.5)
    hi_arr = lo_arr + 1.0
    got = vec(xs, lo_arr, hi_arr)
    expected = np.asarray(
        [CORPUS.clamp(x, lo, hi) for x, lo, hi in zip(xs, lo_arr, hi_arr, strict=True)]
    )
    assert np.allclose(got, expected, equal_nan=True)


@hypothesis.given(arrays(finite))
def test_piecewise_diff(xs: np.ndarray) -> None:
    check(CORPUS.piecewise, xs)


@hypothesis.given(arrays(finite))
def test_ternary_diff(xs: np.ndarray) -> None:
    check(CORPUS.ternary, xs)


def _nonsingular(pair: tuple[float, float]) -> bool:
    return pair[0] * pair[0] + pair[1] * pair[1] > 1e-6


#: lanes where arith_ops' denominator x*x + y*y is not ~0; filtering per
#: lane (not assume() over all 12) keeps Hypothesis from discarding most
#: examples, since it draws 0.0 often
small_pairs: st.SearchStrategy[tuple[float, float]] = st.tuples(small, small)
nonsingular_lanes = st.lists(
    # Hypothesis 6.100's conditional TypeVar definition confuses Pyrefly.
    # Adapt only the callback boundary; the predicate keeps its tuple signature.
    small_pairs.filter(cast(Callable[[Any], bool], _nonsingular)),
    min_size=12,
    max_size=12,
)


@hypothesis.given(nonsingular_lanes)
def test_arith_diff(lanes: list[tuple[float, float]]) -> None:
    xs, ys = (np.asarray(col, dtype=np.float64) for col in zip(*lanes, strict=True))
    check(CORPUS.arith_ops, xs, ys)


@hypothesis.given(arrays(small), arrays(small))
def test_boolop_numeric_diff(xs: np.ndarray, ys: np.ndarray) -> None:
    check(CORPUS.boolop_numeric, xs, ys)


@hypothesis.given(arrays(small), arrays(small))
def test_math_two_arg_diff(xs: np.ndarray, ys: np.ndarray) -> None:
    check(CORPUS.math_two_arg, xs, ys)


@hypothesis.given(arrays(positive))
def test_math_calls_diff(xs: np.ndarray) -> None:
    check(CORPUS.math_calls, xs)


@hypothesis.given(arrays(small))
def test_minmax_diff(xs: np.ndarray) -> None:
    ys = np.asarray([0.5, -1.5, 2.5] * (len(xs) // 3) + [0.0] * (len(xs) % 3))
    check(CORPUS.minmax, xs, ys)


@hypothesis.given(arrays(small))
def test_ssa_rebind_diff(xs: np.ndarray) -> None:
    check(CORPUS.ssa_rebind, xs)


@hypothesis.given(arrays(small), arrays(small))
def test_complex_expr_diff(xs: np.ndarray, ys: np.ndarray) -> None:
    check(CORPUS.complex_expr, xs, ys)


@hypothesis.given(arrays(small))
def test_casts_diff(xs: np.ndarray) -> None:
    check(CORPUS.casts, xs)


@hypothesis.given(arrays(small))
def test_compare_chain_diff(xs: np.ndarray) -> None:
    vec = array_vectorize.vectorize(CORPUS.compare_chain)
    got = vec(xs)
    expected = np.asarray([CORPUS.compare_chain(x) for x in xs])
    assert np.array_equal(got, expected)


@hypothesis.given(arrays(small))
def test_loop_accumulate_diff(xs: np.ndarray) -> None:
    check(CORPUS.loop_accumulate, xs)


@hypothesis.given(arrays(small))
def test_loop_with_branch_diff(xs: np.ndarray) -> None:
    check(CORPUS.loop_with_branch, xs)


@hypothesis.given(arrays(small))
def test_nested_loops_diff(xs: np.ndarray) -> None:
    check(CORPUS.nested_loops, xs)


@hypothesis.given(arrays(small))
def test_helper_outer_diff(xs: np.ndarray) -> None:
    check(CORPUS.helper_outer, xs)


@hypothesis.given(arrays(small))
def test_loop_helper_caller_diff(xs: np.ndarray) -> None:
    check(CORPUS.loop_helper_caller, xs)


@hypothesis.given(arrays(small), arrays(small))
def test_second_oracle_np_vectorize(xs: np.ndarray, ys: np.ndarray) -> None:
    """Differential vs np.vectorize of the original."""
    vec = array_vectorize.vectorize(CORPUS.add)
    oracle = np.vectorize(CORPUS.add)
    assert np.allclose(vec(xs, ys), oracle(xs, ys))


def test_int_dtypes_differential() -> None:
    vec = array_vectorize.vectorize(CORPUS.ssa_rebind)
    for dtype in (np.int32, np.int64):
        xs = np.asarray([-3, 0, 1, 7], dtype=dtype)
        got = vec(xs)
        expected = np.asarray([CORPUS.ssa_rebind(int(x)) for x in xs])
        assert np.allclose(got, expected)


def test_edge_values_relu_ternary() -> None:
    edges = np.asarray([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan, 5e-324, -5e-324])
    vec = array_vectorize.vectorize(CORPUS.relu)
    expected = np.asarray([CORPUS.relu(float(x)) for x in edges])
    assert np.allclose(vec(edges), expected, equal_nan=True)
    vec2 = array_vectorize.vectorize(CORPUS.ternary)
    expected2 = np.asarray([CORPUS.ternary(float(x)) for x in edges])
    assert np.allclose(vec2(edges), expected2, equal_nan=True)


def test_float32_differential() -> None:
    vec = array_vectorize.vectorize(CORPUS.psi)
    xs = np.asarray([-1.5, -0.2, 0.0, 0.3, 2.0], dtype=np.float32)
    got = vec(xs)
    expected = np.asarray([CORPUS.psi(float(x)) for x in xs], dtype=np.float32)
    assert np.allclose(got, expected, equal_nan=True, rtol=1e-6)
