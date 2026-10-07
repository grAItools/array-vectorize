"""Memoization of vectorize(): canonical cache, backrefs, idempotent revectorization."""

from __future__ import annotations

import numpy as np
import pytest

import array_vectorize


def relu_fn(x: float) -> float:
    if x < 0:
        return 0.0
    return x


def sigmoid_like(x: float, alpha: float = 1.0) -> float:
    """A smooth sigmoid-like squashing function.

    Maps ``x`` into ``(0, 1)`` using ``alpha`` as the steepness.
    """
    if x < 0:
        return 0.0
    return 1.0 / (1.0 + alpha * x)


def helper_inner_fn(y: float) -> float:
    return y * y + 1.0


def helper_outer_fn(x: float) -> float:
    return helper_inner_fn(x) + helper_inner_fn(x * 2.0)


def test_backref_on_canonical_vectorization() -> None:
    vec = array_vectorize.vectorize(sigmoid_like)
    assert sigmoid_like.__array_vectorized__ is vec


def test_vectorize_memoized_per_original() -> None:
    assert array_vectorize.vectorize(sigmoid_like) is array_vectorize.vectorize(sigmoid_like)
    assert array_vectorize.vectorize(sigmoid_like) is sigmoid_like.__array_vectorized__


def test_revectorization_returns_same_object() -> None:
    vec = array_vectorize.vectorize(sigmoid_like)
    assert array_vectorize.vectorize(vec) is vec


def test_verify_runs_on_cache_hit() -> None:
    args = (np.asarray([-1.0, 0.5, 2.0]),)
    array_vectorize.vectorize(sigmoid_like, verify=args)
    array_vectorize.vectorize(sigmoid_like, verify=args)


def test_backref_not_overwritten_by_variants() -> None:
    vec = array_vectorize.vectorize(sigmoid_like)
    array_vectorize.vectorize(sigmoid_like, protect_domains=True)
    assert sigmoid_like.__array_vectorized__ is vec
    array_vectorize.vectorize(sigmoid_like, namespace=np)
    assert sigmoid_like.__array_vectorized__ is vec


def test_backref_on_fallback_wrapper() -> None:
    def bad(x: float) -> float:
        while x > 0:
            x = x - 1
        return x

    with pytest.warns(UserWarning, match="falling back"):
        vec = array_vectorize.vectorize(bad, fallback=True)
    assert bad.__array_vectorized__ is vec


def test_helpers_do_not_get_backref() -> None:
    # D7 helper compilations are internal: the user's helper function
    # must not grow an __array_vectorized__ marker
    array_vectorize.vectorize(helper_outer_fn)
    assert not hasattr(helper_inner_fn, "__array_vectorized__")
    assert helper_outer_fn.__array_vectorized__ is not None


def test_idempotent_revectorization() -> None:
    vec1 = array_vectorize.vectorize(relu_fn)
    vec2 = array_vectorize.vectorize(vec1)
    x = np.asarray([-1.0, 2.0])
    assert np.allclose(vec1(x), vec2(x))
    assert "xp.where" in vec2.source
    # canonical calls are memoized per scalar original: re-vectorizing
    # returns the very same object
    assert vec2 is vec1
