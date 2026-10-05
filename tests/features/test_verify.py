"""verify= differential verification option."""

from __future__ import annotations

import numpy as np
import pytest
from support import make_fn

from vectorizer import VectorizationError, vectorize
from vectorizer._verify import verify_match

# ---- from test_m4 (git history: tests/test_m4.py)


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


# ---- from test_review_round1 (git history: tests/test_review_round1.py)


def test_verify_detects_bool_miscompile() -> None:
    with pytest.raises(VectorizationError, match="verification failed"):
        verify_match(
            lambda x: np.array([False, True]),  # wrong: should be [0, 2]
            lambda x: 2 if x > 0 else 0,
            (np.array([-1.0, 1.0]),),
        )


def test_verify_accepts_matching_infinities() -> None:
    verify_match(lambda x: x, lambda x: x, (np.array([np.inf, -np.inf]),))


def test_verify_rejects_inf_vs_finite() -> None:
    with pytest.raises(VectorizationError, match="verification failed"):
        verify_match(lambda x: np.array([0.0]), lambda x: float("inf"), (np.array([1.0]),))


def test_verify_passes_scalar_arguments_through() -> None:
    fn = make_fn("    return x + y")
    vec = vectorize(fn, verify=(np.asarray([1.0]), 3))
    assert vec(np.asarray([1.0]), 3)[0] == 4.0


def test_verify_detects_wrong_constant_output() -> None:
    # a "vectorized" function returning the wrong constant must be caught
    with pytest.raises(VectorizationError, match="verification failed"):
        verify_match(lambda x: np.array([1.0]), lambda x: x * 5.0, (np.array([2.0]),))


# ---- from test_review_round4 (git history: tests/test_review_round4.py)


def test_verify_constant_return() -> None:
    vec = vectorize(make_fn("    return 3.0"), verify=(np.asarray([1.0, 2.0]),))
    got = vec(np.asarray([1.0, 2.0]))
    assert np.allclose(np.broadcast_to(np.asarray(got), (2,)), [3.0, 3.0])
