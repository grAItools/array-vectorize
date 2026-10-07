"""Docstrings on vectorized functions: summary prefix, source round trip, Notes merging."""

from __future__ import annotations

import inspect

import pytest

import array_vectorize
from array_vectorize.codegen import docstring


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


def test_name_and_docstring() -> None:
    vec = array_vectorize.vectorize(relu_fn)
    assert vec.__name__ == "relu_fn"
    doc = inspect.getdoc(vec) or ""
    assert doc.splitlines()[0] == "(array-vectorized) no docstring on the scalar original."
    assert "def relu_fn(x:" in doc
    assert "if x < 0:" in doc


def test_docstring_prefixed_summary() -> None:
    vec = array_vectorize.vectorize(sigmoid_like)
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
    vec = array_vectorize.vectorize(fn)
    doc = inspect.getdoc(vec)
    assert doc is not None
    assert docstring.extract_scalar_source(doc) == inspect.getsource(fn).strip()


def test_docstring_numpy_style_notes() -> None:
    # a NumPy-style original gets a NumPy-style Notes section
    vec = array_vectorize.vectorize(numpy_doc_fn)
    doc = inspect.getdoc(vec)
    assert "Notes\n-----" in (doc or "")


def test_docstring_existing_notes_merged() -> None:
    vec = array_vectorize.vectorize(notes_doc_fn)
    doc = inspect.getdoc(vec) or ""
    # exactly one header line (the copy inside the embedded source is indented)
    assert [line for line in doc.splitlines() if line == "Notes:"] == ["Notes:"]
    assert "Pre-existing note." in doc


def test_docstring_marker_quoting_degrades_to_legacy() -> None:
    # the original's docs quote the marker line, so the combined round
    # trip cannot be guaranteed: the legacy source-only docstring is
    # emitted instead (cleandoc-exact, like the pre-redesign format)
    vec = array_vectorize.vectorize(marker_quoting_fn)
    assert inspect.getdoc(vec) == inspect.getsource(marker_quoting_fn).strip()
