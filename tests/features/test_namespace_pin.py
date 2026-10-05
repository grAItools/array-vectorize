"""namespace= pinning (Strategy A): pinned codegen and call semantics.

A pinned vectorized function binds ``xp`` directly to the provided
namespace instead of extracting it from its arguments; all-scalar calls
become legal.
"""

from __future__ import annotations

from pathlib import Path

import corpus
import numpy as np
import pytest
from support import make_fn

from array_vectorize import vectorize

X = np.asarray([-2.0, -0.5, 0.0, 0.5, 2.0, 10.0])
Y = np.asarray([3.0, 1.5, 1.0, 0.5, -1.0, -3.0])


# ---- pinned source shape


def test_pinned_source_shape() -> None:
    vec = vectorize(corpus.add, namespace=np)
    assert "def add_vec(x, y, *, _namespace=None):" in vec.source
    assert "xp = _namespace" in vec.source
    # pinned generated source has zero imports and no namespace extraction
    assert "import" not in vec.source
    assert "array_namespace" not in vec.source
    assert vec.__kwdefaults__["_namespace"] is np


def test_unpinned_source_unchanged() -> None:
    unpinned = vectorize(corpus.add)
    pinned = vectorize(corpus.add, namespace=np)
    # byte-identical to the pre-pin generated source (the golden snapshot)
    golden = (Path(__file__).parent.parent / "golden" / "cases" / "add.py").read_text()
    assert unpinned.source == golden
    assert pinned.source != unpinned.source
    assert "xp = _namespace" not in unpinned.source
    assert "_namespace=None" not in unpinned.source


def test_pinned_closure_array_still_hidden_param() -> None:
    vec = vectorize(corpus.closure_array, namespace=np)
    assert "*, ARR=None, _namespace=None" in vec.source
    assert vec.__kwdefaults__["_namespace"] is np
    assert np.allclose(vec(np.asarray([0.0, 1.0, 2.0])), [1.0, 3.0, 5.0])


# ---- behavior parity


@pytest.mark.parametrize("name", ["add", "relu", "psi", "piecewise", "math_calls"])
def test_pinned_numpy_matches_unpinned(name: str) -> None:
    fn = getattr(corpus, name)
    args = (X, Y) if name == "add" else (X,)
    assert np.allclose(vectorize(fn, namespace=np)(*args), vectorize(fn)(*args), equal_nan=True)


def test_pinned_strict_matches_unpinned() -> None:
    import array_api_strict as xps

    a = xps.asarray([1.0, -2.0, 3.0])
    unpinned = vectorize(corpus.relu)
    pinned = vectorize(corpus.relu, namespace=xps)
    assert list(map(float, pinned(a))) == list(map(float, unpinned(a)))


# ---- all-scalar calls (impossible unpinned)


def test_all_scalar_call_with_pin() -> None:
    fn = make_fn("    return math.sqrt(x * x)", defaults="x")
    vec = vectorize(fn, namespace=np)
    assert float(vec(3.0)) == pytest.approx(3.0)


def test_all_scalar_minmax_with_pin() -> None:
    fn = make_fn("    return min(x, 3.0) + math.sqrt(x)", defaults="x")
    vec = vectorize(fn, namespace=np)
    assert float(vec(4.0)) == pytest.approx(3.0 + 2.0)


def test_all_scalar_call_without_pin_raises() -> None:
    fn = make_fn("    return math.sqrt(x * x)", defaults="x")
    vec = vectorize(fn)
    with pytest.raises(TypeError, match="array_namespace requires at least one non-scalar"):
        vec(3.0)


# ---- name collisions


def test_user_param_named_namespace_is_kept() -> None:
    fn = make_fn("    return x + _namespace", defaults="x, _namespace")
    vec = vectorize(fn, namespace=np)
    # the user's parameter keeps its name; ours is mangled
    assert "def subject_vec(x, _namespace, *, _namespace_1=None):" in vec.source
    assert "xp = _namespace_1" in vec.source
    assert vec.__kwdefaults__["_namespace_1"] is np
    assert np.allclose(vec(np.asarray([1.0]), np.asarray([5.0])), [6.0])
    # user keyword calls keep working
    assert np.allclose(vec(np.asarray([1.0]), _namespace=np.asarray([5.0])), [6.0])


def test_user_local_named_namespace_is_kept() -> None:
    fn = make_fn("    _namespace = x + 1\n    return _namespace * 2", defaults="x")
    vec = vectorize(fn, namespace=np)
    assert "_namespace = x + 1" in vec.source
    assert "xp = _namespace_1" in vec.source
    assert np.allclose(vec(np.asarray([1.0])), [4.0])
