# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""fallback=True / strict=False element-loop fallback behavior."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import support

import array_vectorize

# --------------------------------------------------------------- fallback=True


def test_fallback_used_on_unsupported() -> None:
    def bad(x: float) -> float:
        while x > 0:
            x = x - 1
        return x

    with pytest.warns(UserWarning, match="falling back"):
        vec = array_vectorize.vectorize(bad, fallback=True)
    xs = np.asarray([5.0, -2.0])
    assert np.allclose(vec(xs), [0.0, -2.0])


def test_fallback_broadcasts_args() -> None:
    def bad(x: float, y: float) -> float:
        while x > 0:
            x = x - y
        return x

    with pytest.warns(UserWarning, match="falling back"):
        vec = array_vectorize.vectorize(bad, fallback=True)
    xs = np.asarray([[3.0]])  # broadcasts against ys
    ys = np.asarray([[1.0, 2.0]])
    assert np.allclose(vec(xs, ys), [[0.0, -1.0]])


def test_fallback_array_kwargs() -> None:
    fn = support.make_fn("    while x < y:\n        x = x + 1\n    return x")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = array_vectorize.vectorize(fn, fallback=True)
    assert np.allclose(v(x=np.asarray([1.0, 3.0])), [2.0, 3.0])
    assert np.allclose(v(np.asarray([1.0, 3.0]), y=np.asarray([2.0, 4.0])), [2.0, 4.0])


def test_fallback_signature_and_source() -> None:
    def bad(x: float, scale: float = 2.0) -> float:
        while x > 0:
            x = x - scale
        return x

    import inspect

    with pytest.warns(UserWarning, match="falling back"):
        vec = array_vectorize.vectorize(bad, fallback=True)
    assert list(inspect.signature(vec).parameters) == ["x", "scale"]
    assert "fallback" in support.with_metadata(vec).source


def test_fallback_non_str_docstring_left_verbatim() -> None:
    # a pathological non-str __doc__ (Python allows arbitrary objects
    # there) is copied verbatim by functools.wraps and must not crash
    # the summary prefixing
    def bad(x: float) -> float:
        while x > 0:
            x = x - 1
        return x

    bad.__doc__ = 42  # type: ignore[assignment]
    with pytest.warns(UserWarning, match="falling back"):
        vec = array_vectorize.vectorize(bad, fallback=True)
    assert getattr(vec, "__doc__") == 42  # noqa: B009 - deliberately non-string metadata
    assert np.allclose(vec(np.asarray([5.0, -2.0])), [0.0, -2.0])


# ------------------------------------ strict=False and the no-fallback default


def test_strict_false_implies_fallback() -> None:
    def bad(x: float) -> float:
        while x > 0:
            x = x - 1
        return x

    with pytest.warns(UserWarning, match="falling back"):
        vec = array_vectorize.vectorize(bad, strict=False)
    assert np.allclose(vec(np.asarray([5.0, -1.0])), [0.0, -1.0])


def test_without_fallback_unsupported_raises() -> None:
    # sanity: an unsupported function without fallback still raises
    def also_bad(x: float) -> float:
        while x > 0:
            x = x - 1
        return x

    with pytest.raises(array_vectorize.VectorizationError):
        array_vectorize.vectorize(also_bad)
