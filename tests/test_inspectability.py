"""Inspectability tests: getsource, linecache, tracebacks, attributes."""

from __future__ import annotations

import ast
import inspect
import linecache
import traceback

import numpy as np

from array_vectorize import get_source, vectorize


def relu_fn(x: float) -> float:
    if x < 0:
        return 0.0
    return x


def test_source_attribute() -> None:
    vec = vectorize(relu_fn)
    assert isinstance(vec.source, str)
    assert "xp.where" in vec.source


def test_getsource_returns_generated_not_original() -> None:
    vec = vectorize(relu_fn)
    src = inspect.getsource(vec)
    assert "array_namespace" in src
    assert "def relu_fn_vec" in src
    assert "def relu_fn(x:" not in src.split('"""')[0]  # only inside docstring


def test_getsourcelines() -> None:
    vec = vectorize(relu_fn)
    lines, lineno = inspect.getsourcelines(vec)
    assert any("array_namespace" in line for line in lines)
    assert lineno >= 1


def test_linecache_survives_checkcache() -> None:
    vec = vectorize(relu_fn)
    filename = vec.__code__.co_filename
    assert filename == "<array_vectorize:relu_fn>"
    linecache.checkcache(filename)
    assert linecache.cache[filename][2]  # still registered
    assert "array_namespace" in "".join(linecache.getlines(filename))


def test_traceback_renders_generated_lines() -> None:
    vec = vectorize(relu_fn)
    try:
        vec(1.0)  # no array among args -> namespace resolution fails
    except TypeError:
        text = traceback.format_exc()
        assert "<array_vectorize:relu_fn>" in text
    else:
        raise AssertionError("expected TypeError for all-scalar call")


def test_wrapped_points_to_original() -> None:
    vec = vectorize(relu_fn)
    # __wrapped__ is deliberately not set (it would hide the generated source
    # from inspect.getsourcelines, which unwraps unconditionally); the
    # original is reachable via the marker and drives the signature.
    assert vec._vectorized_original is relu_fn
    assert inspect.signature(vec) == inspect.signature(relu_fn)


def test_signature_preserved() -> None:
    def with_defaults(x: float, scale: float = 2.0, *, bias: float = 0.5) -> float:
        return x * scale + bias

    vec = vectorize(with_defaults)
    sig = inspect.signature(vec)
    assert list(sig.parameters) == ["x", "scale", "bias"]
    assert sig.parameters["scale"].default == 2.0
    assert sig.parameters["bias"].default == 0.5
    assert sig.parameters["bias"].kind is inspect.Parameter.KEYWORD_ONLY


def test_name_and_docstring() -> None:
    vec = vectorize(relu_fn)
    assert vec.__name__ == "relu_fn"
    assert "def relu_fn(x:" in (vec.__doc__ or "")
    assert "if x < 0:" in (vec.__doc__ or "")


def test_docstring_cleandoc_exact() -> None:
    # the docstring indents the original source to body level; cleandoc
    # (ast.get_docstring / inspect.getdoc) recovers the original verbatim
    vec = vectorize(relu_fn)
    fn_def = ast.parse(vec.source).body[-1]
    assert isinstance(fn_def, ast.FunctionDef)
    original = inspect.getsource(relu_fn).strip()
    assert ast.get_docstring(fn_def) == original
    assert inspect.getdoc(vec) == original


def test_idempotent_revectorization() -> None:
    vec1 = vectorize(relu_fn)
    vec2 = vectorize(vec1)
    x = np.asarray([-1.0, 2.0])
    assert np.allclose(vec1(x), vec2(x))
    assert "xp.where" in vec2.source


def test_get_source_helper() -> None:
    vec = vectorize(relu_fn)
    assert get_source(vec) == vec.source
    try:
        get_source(relu_fn)
    except Exception as exc:
        assert "not a vectorized function" in str(exc)
    else:
        raise AssertionError("expected error for non-vectorized input")


def test_lambda_vectorize() -> None:
    vec = vectorize(lambda x: x * 2.0)
    got = vec(np.asarray([1.0, 2.0]))
    assert np.allclose(got, [2.0, 4.0])
    assert "def lambda_vec" in vec.source


def test_help_renders() -> None:
    # help() must not raise; docstring contains the original source
    vec = vectorize(relu_fn)
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        help(vec)
    assert "relu_fn" in buf.getvalue()
