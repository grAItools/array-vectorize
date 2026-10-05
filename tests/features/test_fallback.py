"""fallback=True / strict=False element-loop fallback behavior."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from support import make_fn

from array_vectorize import VectorizationError, vectorize

# ---- from test_composition (git history: tests/test_composition.py)


def test_fallback_used_on_unsupported() -> None:
    def bad(x: float) -> float:
        while x > 0:
            x = x - 1
        return x

    with pytest.warns(UserWarning, match="falling back"):
        vec = vectorize(bad, fallback=True)
    xs = np.asarray([5.0, -2.0])
    assert np.allclose(vec(xs), [0.0, -2.0])


def test_strict_false_implies_fallback() -> None:
    def bad(x: float) -> float:
        while x > 0:
            x = x - 1
        return x

    with pytest.warns(UserWarning, match="falling back"):
        vec = vectorize(bad, strict=False)
    assert np.allclose(vec(np.asarray([5.0, -1.0])), [0.0, -1.0])


def test_fallback_no_vectorization_error_raised() -> None:
    def bad(x: float) -> float:
        return x

    bad.__code__ = bad.__code__  # keep linters quiet

    # sanity: same function without fallback still raises
    def also_bad(x: float) -> float:
        while x > 0:
            x = x - 1
        return x

    with pytest.raises(VectorizationError):
        vectorize(also_bad)


def test_fallback_broadcasts_args() -> None:
    def bad(x: float, y: float) -> float:
        while x > 0:
            x = x - y
        return x

    with pytest.warns(UserWarning):
        vec = vectorize(bad, fallback=True)
    xs = np.asarray([[3.0]])  # broadcasts against ys
    ys = np.asarray([[1.0, 2.0]])
    assert np.allclose(vec(xs, ys), [[0.0, -1.0]])


def test_fallback_signature_and_source() -> None:
    def bad(x: float, scale: float = 2.0) -> float:
        while x > 0:
            x = x - scale
        return x

    import inspect

    with pytest.warns(UserWarning):
        vec = vectorize(bad, fallback=True)
    assert list(inspect.signature(vec).parameters) == ["x", "scale"]
    assert "fallback" in vec.source


# ---- from test_review_round1 (git history: tests/test_review_round1.py)


def test_fallback_array_kwargs() -> None:
    fn = make_fn("    while x < y:\n        x = x + 1\n    return x")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = vectorize(fn, fallback=True)
    assert np.allclose(v(x=np.asarray([1.0, 3.0])), [2.0, 3.0])
    assert np.allclose(v(np.asarray([1.0, 3.0]), y=np.asarray([2.0, 4.0])), [2.0, 4.0])
