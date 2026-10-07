"""Docstring builder tests: prefixing, style detection, Notes merge, round trip."""

from __future__ import annotations

import inspect

import pytest

from array_vectorize.codegen import docstring, emitter

SOURCE = (
    "def clamp(x, lo=0.0, hi=1.0):\n"
    "    if x < lo:\n"
    "        return lo\n"
    "    if x > hi:\n"
    "        return hi\n"
    "    return x"
)

PLAIN_DOC = "Clamp ``x`` into ``[lo, hi]``.\n\nValues outside are cut off."
GOOGLE_DOC = "Clamp ``x``.\n\nArgs:\n    x: input\n\nReturns:\n    clamped value"
GOOGLE_NOTES_DOC = "Clamp ``x``.\n\nNotes:\n    Pre-existing note."
GOOGLE_NOTES_MIDDLE_DOC = (
    "Clamp ``x``.\n\nNotes:\n    Pre-existing note.\n\nReturns:\n    clamped value"
)
NUMPY_DOC = "Clamp ``x``.\n\nParameters\n----------\nx : float\n    input"
NUMPY_NOTES_DOC = "Clamp ``x``.\n\nNotes\n-----\nPre-existing note."
NUMPY_NOTES_MIDDLE_DOC = (
    "Clamp ``x``.\n\nNotes\n-----\nPre-existing note.\n\nExamples\n--------\n>>> clamp(1)"
)
GOOGLE_FLAT_CONTENT_DOC = "Summary.\n\nNotes:\ncontent at column zero"


def test_prefix_on_summary_line() -> None:
    doc = docstring.build_docstring(GOOGLE_DOC, SOURCE)
    assert doc.splitlines()[0] == f"{docstring.DOCSTRING_PREFIX}Clamp ``x``."
    assert "Args:" in doc


def test_no_docstring_synthesizes_summary() -> None:
    doc = docstring.build_docstring(None, SOURCE)
    assert doc.splitlines()[0] == docstring.DOCSTRING_PREFIX + docstring.NO_DOCSTRING_SUMMARY


@pytest.mark.parametrize("blank", ["", "   ", "\n\n"])
def test_blank_docstring_treated_as_absent(blank: str) -> None:
    assert docstring.build_docstring(blank, SOURCE) == docstring.build_docstring(None, SOURCE)


def test_google_notes_section_shape() -> None:
    doc = docstring.build_docstring(GOOGLE_DOC, SOURCE)
    lines = doc.splitlines()
    assert "Notes:" in lines
    lead = lines[lines.index("Notes:") + 1]
    assert lead == f"    {docstring.LEAD_IN}"
    assert lines[lines.index(lead) + 2] == " " * 8 + "def clamp(x, lo=0.0, hi=1.0):"


def test_numpy_notes_section_shape() -> None:
    doc = docstring.build_docstring(NUMPY_DOC, SOURCE)
    lines = doc.splitlines()
    assert "Notes" in lines
    assert lines[lines.index("Notes") + 1] == "-----"
    assert lines[lines.index("Notes") + 2] == docstring.LEAD_IN
    assert lines[lines.index("Notes") + 4] == " " * 4 + "def clamp(x, lo=0.0, hi=1.0):"


@pytest.mark.parametrize(
    "doc",
    [
        PLAIN_DOC,
        GOOGLE_DOC,
        GOOGLE_NOTES_DOC,
        GOOGLE_NOTES_MIDDLE_DOC,
        GOOGLE_FLAT_CONTENT_DOC,
        NUMPY_DOC,
        NUMPY_NOTES_DOC,
        NUMPY_NOTES_MIDDLE_DOC,
    ],
    ids=[
        "plain",
        "google",
        "google-notes",
        "google-notes-middle",
        "google-flat-content",
        "numpy",
        "numpy-notes",
        "numpy-notes-middle",
    ],
)
def test_round_trip(doc: str) -> None:
    assert docstring.extract_scalar_source(docstring.build_docstring(doc, SOURCE)) == SOURCE


def test_google_notes_merged_not_duplicated() -> None:
    doc = docstring.build_docstring(GOOGLE_NOTES_DOC, SOURCE)
    assert [line for line in doc.splitlines() if line.strip() == "Notes:"] == ["Notes:"]
    # the pre-existing note stays, before our marker
    assert doc.index("Pre-existing note.") < doc.index(docstring.LEAD_IN)


def test_google_notes_in_middle_inserts_before_next_header() -> None:
    doc = docstring.build_docstring(GOOGLE_NOTES_MIDDLE_DOC, SOURCE)
    # our source block lands inside the Notes section, before Returns:
    assert doc.index("def clamp(x") < doc.index("Returns:")
    assert doc.index(docstring.LEAD_IN) < doc.index("Returns:")


def test_numpy_notes_merged_not_duplicated() -> None:
    doc = docstring.build_docstring(NUMPY_NOTES_DOC, SOURCE)
    titles = [i for i, line in enumerate(doc.splitlines()) if line == "Notes"]
    assert len(titles) == 1
    assert doc.index("Pre-existing note.") < doc.index(docstring.LEAD_IN)


def test_numpy_notes_in_middle_inserts_before_next_header() -> None:
    doc = docstring.build_docstring(NUMPY_NOTES_MIDDLE_DOC, SOURCE)
    assert doc.index("def clamp(x") < doc.index("Examples")
    assert doc.index(docstring.LEAD_IN) < doc.index("Examples")


def test_google_flat_content_gets_default_indent() -> None:
    # malformed section content at the header's own indent: the block
    # still lands at the default content indent and round-trips
    doc = docstring.build_docstring(GOOGLE_FLAT_CONTENT_DOC, SOURCE)
    lead = next(line for line in doc.splitlines() if line.strip() == docstring.LEAD_IN)
    assert lead == f"    {docstring.LEAD_IN}"


def test_empty_section_body_gets_default_indent() -> None:
    # an empty Notes section (header, then the next header): our block
    # still lands inside it at the default content indent
    doc = docstring.build_docstring("Summary.\n\nNotes:\n\nExamples:\n    stuff", SOURCE)
    lines = doc.splitlines()
    notes_idx = lines.index("Notes:")
    assert lines[notes_idx + 1] == f"    {docstring.LEAD_IN}"
    assert doc.index("def clamp(x") < doc.index("Examples:")


@pytest.mark.parametrize(
    ("doc", "expected"),
    [
        (NUMPY_DOC, True),
        (NUMPY_NOTES_DOC, True),
        (GOOGLE_DOC, False),
        (PLAIN_DOC, False),
        ("Some text\n----\nnot a section title", False),
        ("Args:\n    x: google style", False),
        ("Parameters\n=\nshort underline", False),
        ("Parameters\n--\ntoo short", False),
    ],
    ids=[
        "numpy-parameters",
        "numpy-notes",
        "google",
        "plain",
        "dash-under-prose",
        "google-args",
        "equals-not-dashes",
        "underline-too-short",
    ],
)
def test_is_numpy_style(doc: str, expected: bool) -> None:
    assert docstring.is_numpy_style(doc) is expected


def test_extract_no_marker_raises() -> None:
    with pytest.raises(ValueError, match="no array-vectorize source marker"):
        docstring.extract_scalar_source("Just a docstring without a marker.")


def test_extract_empty_block_raises() -> None:
    with pytest.raises(ValueError, match="no source block"):
        docstring.extract_scalar_source(
            f"Summary.\n\n{docstring.LEAD_IN}\n\nUnindented tail at column zero."
        )


def test_prefixed_summary_accepts_raw_docstrings() -> None:
    # the fallback wrapper passes func.__doc__ verbatim (leading newline,
    # function-body indentation) — cleandoc normalizes before prefixing
    raw = "\n    Summary line.\n\n    Body.\n    "
    assert (
        docstring.prefixed_summary(raw).splitlines()[0]
        == f"{docstring.DOCSTRING_PREFIX}Summary line."
    )
    assert (
        docstring.prefixed_summary(None)
        == docstring.DOCSTRING_PREFIX + docstring.NO_DOCSTRING_SUMMARY
    )
    assert (
        docstring.prefixed_summary("")
        == docstring.DOCSTRING_PREFIX + docstring.NO_DOCSTRING_SUMMARY
    )


def test_docstring_value_self_check_passes() -> None:
    value = emitter._docstring_value(GOOGLE_DOC, SOURCE)
    doc = inspect.cleandoc(value)
    assert doc.splitlines()[0] == f"{docstring.DOCSTRING_PREFIX}Clamp ``x``."
    assert docstring.extract_scalar_source(doc) == SOURCE


def test_docstring_value_degrades_on_marker_collision() -> None:
    # the original's docs quote the marker line before our section: the
    # extraction would grab the quoted block, so the legacy source-only
    # docstring is emitted instead
    quoting_doc = f"Summary.\n\n{docstring.LEAD_IN}\n\n    quoted block"
    value = emitter._docstring_value(quoting_doc, SOURCE)
    assert inspect.cleandoc(value) == SOURCE


def test_docstring_value_degrades_on_tab_indented_source() -> None:
    # cleandoc expands tabs on read, so a tab-indented source cannot
    # round-trip through getdoc byte-exactly (the legacy format has the
    # same pre-existing caveat): degrade to the legacy source-only
    # docstring, which still embeds the source text verbatim
    tab_source = "def f(x):\n\treturn x"
    value = emitter._docstring_value(None, tab_source)
    assert docstring.LEAD_IN not in value  # legacy format, not the combined one
    assert "def f(x):\n    \treturn x" in value
