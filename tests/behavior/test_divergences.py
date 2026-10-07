"""Documented divergences between scalar Python semantics and vectorized backends."""

from __future__ import annotations

import numpy as np
import support

import array_vectorize

# --------------------------------------- numeric exceptions become IEEE values


def test_divergence_div_by_zero_is_inf() -> None:
    with np.errstate(divide="ignore"):
        got = array_vectorize.vectorize(lambda x: 1.0 / x)(np.asarray([-1.0, 0.0, 2.0]))
    assert np.allclose(got, [-1.0, np.inf, 0.5])


def test_divergence_neg_sqrt_is_nan() -> None:
    with np.errstate(invalid="ignore"):
        got = support.vfn("cse_opportunity")(np.asarray([-4.0]))
    assert np.isnan(got[0])


def test_remainder_zero_is_nan() -> None:
    fn = support.make_fn("    return x % 0.0")
    with np.errstate(invalid="ignore", divide="ignore"):
        got = array_vectorize.vectorize(fn)(np.asarray([1.0, -1.0]))
    assert np.all(np.isnan(got))


def test_int_floordiv_zero_backend_defined() -> None:
    fn = support.make_fn("    return x // 0")
    with np.errstate(divide="ignore", invalid="ignore"):
        got = array_vectorize.vectorize(fn)(np.asarray([3, -3], dtype=np.int64))
    # NumPy: 0 with a warning; documented as backend-defined (docs/architecture.md, D3)
    assert list(got) == [0, 0]


def test_pow_negative_base_fractional_exponent_is_nan() -> None:
    fn = support.make_fn("    return x ** 0.5")
    with np.errstate(invalid="ignore"):
        got = array_vectorize.vectorize(fn)(np.asarray([-4.0, 4.0]))
    assert np.isnan(got[0])
    assert got[1] == 2.0


# ---------------------------------------------- eager evaluation and promotion


def test_and_or_eager_with_nan_lanes() -> None:
    fn = support.make_fn("    return (x > 0) and (y > 0)")
    vec = array_vectorize.vectorize(fn)
    xs = np.asarray([1.0, -1.0, np.nan])
    ys = np.asarray([1.0, 1.0, 1.0])
    # NaN comparisons: NaN > 0 is False (no exception in vectorized code)
    assert np.array_equal(vec(xs, ys), [True, False, False])


def test_min_max_nan_divergence_documented() -> None:
    # Python: min(a, nan) returns a (nan < a is False); xp.minimum
    # propagates NaN. The documented mapping (left-fold xp.minimum) makes
    # vectorized min NaN-propagating - a documented divergence.
    fn = support.make_fn("    return min(x, y)")
    vec = array_vectorize.vectorize(fn)
    xs = np.asarray([1.0, 1.0])
    ys = np.asarray([np.nan, 2.0])
    got = vec(xs, ys)
    assert np.isnan(got[0])  # xp.minimum(1, nan) -> nan (scalar: 1.0)
    assert got[1] == 1.0


def test_where_branch_type_promotion_divergence() -> None:
    # xp.where promotes branch dtypes: a bool branch merged with a float
    # branch becomes float, while Python's ternary returns the taken branch
    # unpromoted. Values on live lanes are unchanged, but downstream ops
    # can observe the type (here: int remainder-by-zero vs float NaN).
    fn = support.make_fn("    return 0 % (x if x > 1 else x != x)")
    vec = array_vectorize.vectorize(fn)
    xs = np.asarray([0.5])  # takes the bool branch -> 0 % False
    # scalar Python: ZeroDivisionError; vectorized: float 0.0 % 0.0 -> NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        assert np.isnan(vec(xs)[0])


def test_round_negative_zero_divergence() -> None:
    # Python round(-0.0) returns int 0 (sign dropped); xp.round keeps -0.0,
    # which atan2 can observe. Documented mapping divergence.
    fn = support.make_fn("    return math.atan2(round(x), x)")
    vec = array_vectorize.vectorize(fn)
    xs = np.asarray([-0.0])
    assert vec(xs)[0] == -np.pi  # scalar Python: atan2(0, -0.0) = +pi
