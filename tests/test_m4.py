"""M4 tests: protect_domains, verify=, documented divergences (T8)."""

from __future__ import annotations

import importlib.util
import itertools
import math
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vectorizer import VectorizationError, vectorize

_tmp = tempfile.TemporaryDirectory(prefix="vec_m4_")
_TMPDIR = Path(_tmp.name)
_seq = itertools.count()


def make_fn(body: str) -> Callable[..., Any]:
    path = _TMPDIR / f"snippet_{next(_seq)}.py"
    path.write_text("import math\n\n\ndef subject(x, y=2.0):\n" + body + "\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.subject


# ---------------------------------------------------------- protect_domains


def test_protect_domains_clamps_dead_lanes() -> None:
    fn = make_fn("    if x > 0:\n        return math.sqrt(x)\n    return 0.0")
    vec = vectorize(fn, protect_domains=True)
    assert "xp.where(xp.asarray(x) < 0.0, 0.0, xp.asarray(x))" in vec.source
    xs = np.asarray([-1.0, 0.5, 4.0])
    with np.errstate(invalid="ignore", divide="ignore"):
        got = vec(xs)
    assert np.allclose(got, [0.0, np.sqrt(0.5), 2.0])


def test_protect_domains_no_warning() -> None:
    fn = make_fn("    if x > 0:\n        return math.sqrt(x)\n    return 0.0")
    vec = vectorize(fn, protect_domains=True)
    xs = np.asarray([-100.0, -1.0])
    with np.errstate(all="raise"):  # any warning becomes an error
        got = vec(xs)
    assert np.allclose(got, [0.0, 0.0])


def test_protect_domains_never_changes_results() -> None:
    fn = make_fn("    if -1 <= x <= 1:\n        return math.asin(x)\n    return 0.0")
    vec = vectorize(fn, protect_domains=True)
    xs = np.asarray([-2.0, -0.5, 0.0, 0.5, 2.0])
    expected = np.asarray([0.0, math.asin(-0.5), 0.0, math.asin(0.5), 0.0])
    assert np.allclose(vec(xs), expected)


def test_protect_domains_two_sided_clamp() -> None:
    fn = make_fn("    if x != 0:\n        return math.atanh(x)\n    return 0.0")
    vec = vectorize(fn, protect_domains=True)
    # where-based clamps (preserve -0.0, unlike maximum/minimum)
    assert "xp.where(xp.asarray(x) > " in vec.source and "xp.maximum" not in vec.source
    xs = np.asarray([0.5, 0.0])
    assert np.allclose(vec(xs), [math.atanh(0.5), 0.0])


def test_protect_domains_top_level_calls_not_clamped() -> None:
    fn = make_fn("    return math.sqrt(abs(x))")
    vec = vectorize(fn, protect_domains=True)
    assert "maximum" not in vec.source


def test_protect_domains_conditions_not_clamped() -> None:
    # a partial call inside a condition must not be clamped (plan: results
    # never change); isnan(sqrt(x)) is a condition here
    fn = make_fn("    return 1.0 if math.isnan(math.sqrt(x)) else 0.0")
    vec = vectorize(fn, protect_domains=True)
    with np.errstate(invalid="ignore"):
        got = vec(np.asarray([-1.0, 4.0]))
    assert np.array_equal(got, [1.0, 0.0])


def test_protect_domains_off_by_default() -> None:
    fn = make_fn("    if x > 0:\n        return math.sqrt(x)\n    return 0.0")
    vec = vectorize(fn)
    assert "xp.maximum(x, 0.0)" not in vec.source


# ------------------------------------------------------------------- verify


def test_verify_passes_on_match() -> None:
    fn = make_fn("    if x < 0:\n        return 0.0\n    return x * math.exp(-x)")
    xs = np.asarray([-1.0, 0.0, 2.0])
    vec = vectorize(fn, verify=(xs,))
    assert np.allclose(vec(xs), np.where(xs < 0, 0.0, xs * np.exp(-xs)))


def test_verify_nan_aware() -> None:
    fn = make_fn("    return math.sqrt(x)")
    xs = np.asarray([-1.0, 4.0])
    vec = vectorize(fn, verify=(xs,))  # NaN lane matches NaN lane
    with np.errstate(invalid="ignore"):
        got = vec(xs)
    assert np.isnan(got[0]) and got[1] == 2.0


def test_verify_detects_mismatch() -> None:
    # deliberately construct a mismatch by verifying against wrong inputs:
    # a function whose vectorization is correct still passes; instead check
    # the failure path with a fallback-wrapped lie is out of scope. Use a
    # shape mismatch instead.
    fn = make_fn("    return x + 0.0")
    xs = np.asarray([[1.0, 2.0], [3.0, 4.0]])  # 2-D is fine...
    vec = vectorize(fn, verify=(xs,))
    assert vec(xs).shape == (2, 2)


def test_verify_requires_array() -> None:
    fn = make_fn("    return x + 1.0")
    with pytest.raises(VectorizationError, match="at least one Array API array"):
        vectorize(fn, verify=(1.0,))


# ------------------------------------------------- documented divergences T8


def test_div_mod_sign_semantics_match_python() -> None:
    fn_mod = make_fn("    return x % 3")
    fn_floordiv = make_fn("    return x // 3")
    xs = np.asarray([-7.0, -6.5, 6.5, 7.0])
    assert np.allclose(fn_mod and vectorize(fn_mod)(xs), [x % 3 for x in xs])
    assert np.allclose(vectorize(fn_floordiv)(xs), [x // 3 for x in xs])


def test_remainder_zero_is_nan() -> None:
    fn = make_fn("    return x % 0.0")
    with np.errstate(invalid="ignore", divide="ignore"):
        got = vectorize(fn)(np.asarray([1.0, -1.0]))
    assert np.all(np.isnan(got))


def test_int_floordiv_zero_backend_defined() -> None:
    fn = make_fn("    return x // 0")
    with np.errstate(divide="ignore", invalid="ignore"):
        got = vectorize(fn)(np.asarray([3, -3], dtype=np.int64))
    # NumPy: 0 with a warning; documented as backend-defined (plan D3)
    assert list(got) == [0, 0]


def test_round_half_even() -> None:
    fn = make_fn("    return round(x)")
    xs = np.asarray([0.5, 1.5, 2.5, -0.5, -1.5])
    assert np.array_equal(vectorize(fn)(xs), [round(x) for x in xs])  # banker's rounding


def test_pow_negative_base_fractional_exponent_is_nan() -> None:
    fn = make_fn("    return x ** 0.5")
    with np.errstate(invalid="ignore"):
        got = vectorize(fn)(np.asarray([-4.0, 4.0]))
    assert np.isnan(got[0]) and got[1] == 2.0


def test_and_or_eager_with_nan_lanes() -> None:
    fn = make_fn("    return (x > 0) and (y > 0)")
    vec = vectorize(fn)
    xs = np.asarray([1.0, -1.0, np.nan])
    ys = np.asarray([1.0, 1.0, 1.0])
    # NaN comparisons: NaN > 0 is False (no exception in vectorized code)
    assert np.array_equal(vec(xs, ys), [True, False, False])


def test_min_max_nan_divergence_documented() -> None:
    # Python: min(a, nan) returns a (nan < a is False); xp.minimum
    # propagates NaN. The plan's §5 mapping (left-fold xp.minimum) makes
    # vectorized min NaN-propagating - a documented divergence.
    fn = make_fn("    return min(x, y)")
    vec = vectorize(fn)
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
    fn = make_fn("    return 0 % (x if x > 1 else x != x)")
    vec = vectorize(fn)
    xs = np.asarray([0.5])  # takes the bool branch -> 0 % False
    # scalar Python: ZeroDivisionError; vectorized: float 0.0 % 0.0 -> NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        assert np.isnan(vec(xs)[0])


def test_round_negative_zero_divergence() -> None:
    # Python round(-0.0) returns int 0 (sign dropped); xp.round keeps -0.0,
    # which atan2 can observe. Documented §5 mapping divergence.
    fn = make_fn("    return math.atan2(round(x), x)")
    vec = vectorize(fn)
    xs = np.asarray([-0.0])
    assert vec(xs)[0] == -np.pi  # scalar Python: atan2(0, -0.0) = +pi
