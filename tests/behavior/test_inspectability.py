"""Inspectability of vectorized functions: getsource, linecache, tracebacks, help()."""

from __future__ import annotations

import inspect
import linecache
import pickle
import traceback

import pytest
import support

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


def test_source_attribute() -> None:
    vec = array_vectorize.vectorize(relu_fn)
    assert isinstance(support.with_metadata(vec).source, str)
    assert "xp.where" in support.with_metadata(vec).source


def test_getsource_returns_generated_not_original() -> None:
    vec = array_vectorize.vectorize(relu_fn)
    src = inspect.getsource(vec)
    assert "array_namespace" in src
    assert "def relu_fn_vec" in src
    assert "def relu_fn(x:" not in src.split('"""')[0]  # only inside docstring


def test_getsourcelines() -> None:
    vec = array_vectorize.vectorize(relu_fn)
    lines, lineno = inspect.getsourcelines(vec)
    assert any("array_namespace" in line for line in lines)
    assert lineno >= 1


def test_linecache_survives_checkcache() -> None:
    vec = array_vectorize.vectorize(relu_fn)
    filename = vec.__code__.co_filename
    assert filename == "<array_vectorize:relu_fn>"
    linecache.checkcache(filename)
    entry = linecache.cache[filename]
    assert len(entry) == 4  # populated entry, rather than the one-item lazy entry
    assert entry[2]  # still registered
    assert "array_namespace" in "".join(linecache.getlines(filename))


def test_traceback_renders_generated_lines() -> None:
    vec = array_vectorize.vectorize(relu_fn)
    try:
        vec(1.0)  # no array among args -> namespace resolution fails
    except TypeError:
        text = traceback.format_exc()
        assert "<array_vectorize:relu_fn>" in text
    else:
        raise AssertionError("expected TypeError for all-scalar call")


def test_wrapped_points_to_original() -> None:
    vec = array_vectorize.vectorize(relu_fn)
    # __wrapped__ is deliberately not set (it would hide the generated source
    # from inspect.getsourcelines, which unwraps unconditionally); the
    # original is reachable via the marker and drives the signature.
    assert support.with_metadata(vec)._vectorized_original is relu_fn
    assert inspect.signature(vec) == inspect.signature(relu_fn)


def test_module_and_qualname_from_original() -> None:
    # module attribution makes pydoc render "Help on function f in module m"
    # and is safe: getsource/getsourcelines use co_filename + linecache
    vec = array_vectorize.vectorize(relu_fn)
    assert vec.__module__ == relu_fn.__module__
    assert vec.__qualname__ == relu_fn.__qualname__


def test_pickle_fails_loudly_never_aliases() -> None:
    # __module__/__qualname__ match the original, so pickle resolves the
    # name and fails the identity check — it cannot silently serialize
    # (and unpickle to) the scalar original
    vec = array_vectorize.vectorize(relu_fn)
    with pytest.raises(pickle.PicklingError):
        pickle.dumps(vec)


def test_signature_preserved() -> None:
    def with_defaults(x: float, scale: float = 2.0, *, bias: float = 0.5) -> float:
        return x * scale + bias

    vec = array_vectorize.vectorize(with_defaults)
    sig = inspect.signature(vec)
    assert list(sig.parameters) == ["x", "scale", "bias"]
    assert sig.parameters["scale"].default == 2.0
    assert sig.parameters["bias"].default == 0.5
    assert sig.parameters["bias"].kind is inspect.Parameter.KEYWORD_ONLY


def test_get_source_helper() -> None:
    vec = array_vectorize.vectorize(relu_fn)
    assert array_vectorize.get_source(vec) == support.with_metadata(vec).source
    with pytest.raises(array_vectorize.VectorizationError, match="not a vectorized function"):
        array_vectorize.get_source(relu_fn)


def test_help_renders() -> None:
    # help() must not raise; it shows the module header, the prefixed
    # summary, and the original source in the Notes section
    vec = array_vectorize.vectorize(sigmoid_like)
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        help(vec)
    out = buf.getvalue()
    assert f"Help on function sigmoid_like in module {sigmoid_like.__module__}:" in out
    assert "(array-vectorized) A smooth sigmoid-like squashing function." in out
    assert "Notes:" in out
    assert "def sigmoid_like(x: float, alpha: float = 1.0) -> float:" in out
