"""Inspectability tests: getsource, linecache, tracebacks, attributes."""

from __future__ import annotations

import inspect
import linecache
import pickle
import traceback

import numpy as np
import pytest

import array_vectorize
from array_vectorize import VectorizationError, get_source, vectorize
from array_vectorize.codegen.docstring import extract_scalar_source


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


def numpy_doc_fn(x: float) -> float:
    """Square with a NumPy-style docstring.

    Parameters
    ----------
    x : float
        input value

    Returns
    -------
    float
        the square
    """
    return x * x


def notes_doc_fn(x: float) -> float:
    """Clamp nonnegative values.

    Notes:
        Pre-existing note.
    """
    if x < 0:
        return 0.0
    return x


def marker_quoting_fn(x: float) -> float:
    """Quotes the array-vectorize marker in its own docs.

    Vectorized by array-vectorize from this scalar original::

        not the real source
    """
    return x * 3.0


def helper_inner_fn(y: float) -> float:
    return y * y + 1.0


def helper_outer_fn(x: float) -> float:
    return helper_inner_fn(x) + helper_inner_fn(x * 2.0)


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


def test_module_and_qualname_from_original() -> None:
    # module attribution makes pydoc render "Help on function f in module m"
    # and is safe: getsource/getsourcelines use co_filename + linecache
    vec = vectorize(relu_fn)
    assert vec.__module__ == relu_fn.__module__
    assert vec.__qualname__ == relu_fn.__qualname__


def test_pickle_fails_loudly_never_aliases() -> None:
    # __module__/__qualname__ match the original, so pickle resolves the
    # name and fails the identity check — it cannot silently serialize
    # (and unpickle to) the scalar original
    vec = vectorize(relu_fn)
    with pytest.raises(pickle.PicklingError):
        pickle.dumps(vec)


def test_backref_on_canonical_vectorization() -> None:
    vec = vectorize(sigmoid_like)
    assert sigmoid_like.__array_vectorized__ is vec


def test_vectorize_memoized_per_original() -> None:
    assert vectorize(sigmoid_like) is vectorize(sigmoid_like)
    assert vectorize(sigmoid_like) is sigmoid_like.__array_vectorized__


def test_revectorization_returns_same_object() -> None:
    vec = vectorize(sigmoid_like)
    assert vectorize(vec) is vec


def test_verify_runs_on_cache_hit() -> None:
    args = (np.asarray([-1.0, 0.5, 2.0]),)
    vectorize(sigmoid_like, verify=args)
    vectorize(sigmoid_like, verify=args)


def test_backref_not_overwritten_by_variants() -> None:
    vec = vectorize(sigmoid_like)
    vectorize(sigmoid_like, protect_domains=True)
    assert sigmoid_like.__array_vectorized__ is vec
    vectorize(sigmoid_like, namespace=np)
    assert sigmoid_like.__array_vectorized__ is vec


def test_backref_on_fallback_wrapper() -> None:
    def bad(x: float) -> float:
        while x > 0:
            x = x - 1
        return x

    with pytest.warns(UserWarning, match="falling back"):
        vec = vectorize(bad, fallback=True)
    assert bad.__array_vectorized__ is vec


def test_helpers_do_not_get_backref() -> None:
    # D7 helper compilations are internal: the user's helper function
    # must not grow an __array_vectorized__ marker
    vectorize(helper_outer_fn)
    assert not hasattr(helper_inner_fn, "__array_vectorized__")
    assert helper_outer_fn.__array_vectorized__ is not None


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
    doc = inspect.getdoc(vec) or ""
    assert doc.splitlines()[0] == "(array-vectorized) no docstring on the scalar original."
    assert "def relu_fn(x:" in doc
    assert "if x < 0:" in doc


def test_docstring_prefixed_summary() -> None:
    vec = vectorize(sigmoid_like)
    doc = inspect.getdoc(vec)
    assert doc is not None
    assert doc.splitlines()[0] == "(array-vectorized) A smooth sigmoid-like squashing function."
    assert "Maps ``x`` into ``(0, 1)`` using ``alpha`` as the steepness." in doc


@pytest.mark.parametrize(
    "fn", [relu_fn, sigmoid_like, numpy_doc_fn, notes_doc_fn], ids=lambda f: f.__name__
)
def test_docstring_source_round_trip(fn) -> None:
    # the Notes section embeds the scalar source verbatim; cleandoc of the
    # generated docstring recovers it exactly
    vec = vectorize(fn)
    doc = inspect.getdoc(vec)
    assert doc is not None
    assert extract_scalar_source(doc) == inspect.getsource(fn).strip()


def test_docstring_numpy_style_notes() -> None:
    # a NumPy-style original gets a NumPy-style Notes section
    vec = vectorize(numpy_doc_fn)
    doc = inspect.getdoc(vec)
    assert "Notes\n-----" in (doc or "")


def test_docstring_existing_notes_merged() -> None:
    vec = vectorize(notes_doc_fn)
    doc = inspect.getdoc(vec) or ""
    # exactly one header line (the copy inside the embedded source is indented)
    assert [line for line in doc.splitlines() if line == "Notes:"] == ["Notes:"]
    assert "Pre-existing note." in doc


def test_docstring_marker_quoting_degrades_to_legacy() -> None:
    # the original's docs quote the marker line, so the combined round
    # trip cannot be guaranteed: the legacy source-only docstring is
    # emitted instead (cleandoc-exact, like the pre-redesign format)
    vec = vectorize(marker_quoting_fn)
    assert inspect.getdoc(vec) == inspect.getsource(marker_quoting_fn).strip()


def test_idempotent_revectorization() -> None:
    vec1 = vectorize(relu_fn)
    vec2 = vectorize(vec1)
    x = np.asarray([-1.0, 2.0])
    assert np.allclose(vec1(x), vec2(x))
    assert "xp.where" in vec2.source
    # canonical calls are memoized per scalar original: re-vectorizing
    # returns the very same object
    assert vec2 is vec1


def test_get_source_helper() -> None:
    vec = vectorize(relu_fn)
    assert get_source(vec) == vec.source
    with pytest.raises(VectorizationError, match="not a vectorized function"):
        get_source(relu_fn)


def test_lambda_vectorize() -> None:
    vec = vectorize(lambda x: x * 2.0)
    got = vec(np.asarray([1.0, 2.0]))
    assert np.allclose(got, [2.0, 4.0])
    assert "def lambda_vec" in vec.source


def test_help_renders() -> None:
    # help() must not raise; it shows the module header, the prefixed
    # summary, and the original source in the Notes section
    vec = vectorize(sigmoid_like)
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        help(vec)
    out = buf.getvalue()
    assert f"Help on function sigmoid_like in module {sigmoid_like.__module__}:" in out
    assert "(array-vectorized) A smooth sigmoid-like squashing function." in out
    assert "Notes:" in out
    assert "def sigmoid_like(x: float, alpha: float = 1.0) -> float:" in out


def test_package_exposes_version() -> None:
    assert array_vectorize.__version__
