"""Docstring assembly for generated functions.

The generated function's docstring carries the original's documentation
— the summary line prefixed with ``(array-vectorized)`` — plus the
verbatim scalar source in a ``Notes:`` section: Google style by default,
NumPy style when the original's docstring uses NumPy-style section
headers. The section's lead-in line doubles as the marker that
:func:`extract_scalar_source` recovers the source from; codegen
self-checks that round trip and degrades to the legacy source-only
docstring when it cannot be guaranteed. That is docstring presentation
only — compiled semantics are untouched, so design D10
(reject over miscompile) does not apply.
"""

from __future__ import annotations

import inspect

__all__ = [
    "DOCSTRING_PREFIX",
    "LEAD_IN",
    "NO_DOCSTRING_SUMMARY",
    "build_docstring",
    "extract_scalar_source",
    "is_numpy_style",
    "prefixed_summary",
]

#: summary-line prefix marking a vectorized function's docstring
DOCSTRING_PREFIX = "(array-vectorized) "
#: summary line synthesized for originals without a docstring
NO_DOCSTRING_SUMMARY = "no docstring on the scalar original."
#: lead-in of the source section; it ends a reST paragraph with ``::``,
#: so the indented source below renders as a literal block
LEAD_IN = "Vectorized by array-vectorize from this scalar original::"

#: NumPydoc's canonical section titles
_NUMPY_SECTIONS = frozenset(
    {
        "Attributes",
        "Examples",
        "Methods",
        "Notes",
        "Other Parameters",
        "Parameters",
        "Raises",
        "Receives",
        "References",
        "Returns",
        "See Also",
        "Warns",
        "Warnings",
        "Yields",
    }
)

#: Napoleon's Google-style section names (lowercased)
_GOOGLE_SECTIONS = frozenset(
    {
        "args",
        "arguments",
        "attributes",
        "example",
        "examples",
        "keyword args",
        "keyword arguments",
        "methods",
        "note",
        "notes",
        "other parameters",
        "parameters",
        "receive",
        "receives",
        "raise",
        "raises",
        "return",
        "returns",
        "warn",
        "warns",
        "warning",
        "warnings",
        "yield",
        "yields",
    }
)


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def _indent_block(text: str, width: int) -> str:
    pad = " " * width
    return "\n".join(pad + line if line.strip() else "" for line in text.splitlines())


def _numpy_title_at(lines: list[str], i: int) -> bool:
    """True when ``lines[i]`` is a NumPy-style section title (dash underline)."""
    if i + 1 >= len(lines):
        return False
    title, underline = lines[i], lines[i + 1]
    mark = underline.strip()
    return (
        title.strip() in _NUMPY_SECTIONS
        and len(mark) >= 3
        and set(mark) == {"-"}
        and _indent_of(title) == _indent_of(underline)
    )


def is_numpy_style(doc: str) -> bool:
    """Detect NumPy-style section headers (title + dash underline).

    Heuristic by design: mixed-style docstrings count as NumPy (the
    detected style wins), and a plain dash separator under a line that is
    not a section title does not count.
    """
    lines = doc.split("\n")
    return any(_numpy_title_at(lines, i) for i in range(len(lines) - 1))


def _google_section_name(line: str) -> str | None:
    stripped = line.strip()
    if stripped.endswith(":"):
        name = stripped[:-1].lower()
        if name in _GOOGLE_SECTIONS:
            return name
    return None


def _prefixed_head(doc: str) -> str:
    # ``doc`` is cleandoc'd and non-blank
    first, _, rest = doc.partition("\n")
    head = DOCSTRING_PREFIX + first
    if rest.strip():
        head += "\n" + rest.rstrip()
    return head


def prefixed_summary(docstring: str | None) -> str:
    """Prefixed docstring head: summary + body, or the synthesized line.

    Accepts raw docstrings (cleandoc is applied here, and is idempotent
    on already-cleaned input). Used by the fallback wrapper, which has
    only the original's ``__doc__`` available.
    """
    doc = inspect.cleandoc(docstring or "")
    if not doc:
        return DOCSTRING_PREFIX + NO_DOCSTRING_SUMMARY
    return _prefixed_head(doc)


def _new_google_notes(source: str) -> str:
    return f"Notes:\n    {LEAD_IN}\n\n" + _indent_block(source, 8)


def _new_numpy_notes(source: str, indent: int = 0) -> str:
    pad = " " * indent
    return f"{pad}Notes\n{pad}-----\n{pad}{LEAD_IN}\n\n" + _indent_block(source, indent + 4)


def _source_block_lines(source: str, indent: int) -> list[str]:
    return [f"{' ' * indent}{LEAD_IN}", "", *_indent_block(source, indent + 4).split("\n")]


def _merge_google_notes(doc: str, source: str) -> str:
    """Append the source block into an existing ``Notes:`` section, or add one."""
    lines = doc.split("\n")
    header_idx = next(
        (i for i, line in enumerate(lines) if _google_section_name(line) == "notes"), None
    )
    if header_idx is None:
        return doc + "\n\n" + _new_google_notes(source)
    header_indent = _indent_of(lines[header_idx])
    end = len(lines)
    for i in range(header_idx + 1, len(lines)):
        if _google_section_name(lines[i]) is not None and _indent_of(lines[i]) == header_indent:
            end = i
            break
    body = lines[header_idx + 1 : end]
    while body and not body[-1].strip():
        body.pop()
    content_indent = next((_indent_of(line) for line in body if line.strip()), header_indent + 4)
    if content_indent <= header_indent:
        content_indent = header_indent + 4
    tail = lines[end:]
    block = _source_block_lines(source, content_indent)
    new_body = body + ([""] if body else []) + block + ([""] if tail else [])
    return "\n".join([*lines[: header_idx + 1], *new_body, *tail])


def _merge_numpy_notes(doc: str, source: str) -> str:
    """Append the source block into an existing NumPy ``Notes`` section, or add one."""
    lines = doc.split("\n")
    header_idx = next(
        (
            i
            for i in range(len(lines) - 1)
            if lines[i].strip() == "Notes" and _numpy_title_at(lines, i)
        ),
        None,
    )
    if header_idx is None:
        return doc + "\n\n" + _new_numpy_notes(source)
    title_indent = _indent_of(lines[header_idx])
    end = len(lines)
    for i in range(header_idx + 2, len(lines)):
        if _numpy_title_at(lines, i) and _indent_of(lines[i]) == title_indent:
            end = i
            break
    body = lines[header_idx + 2 : end]
    while body and not body[-1].strip():
        body.pop()
    tail = lines[end:]
    block = _source_block_lines(source, title_indent)
    new_body = body + ([""] if body else []) + block + ([""] if tail else [])
    return "\n".join([*lines[: header_idx + 2], *new_body, *tail])


def build_docstring(docstring: str | None, source: str) -> str:
    """Build the generated function's docstring text (logical column 0).

    ``docstring`` is the original's docstring (raw or cleandoc'd; blank or
    ``None`` when absent); ``source`` is the dedented, stripped original
    source. The result's first line is the prefixed summary (or the
    synthesized line), and :func:`extract_scalar_source` recovers
    ``source`` from it exactly.
    """
    doc = inspect.cleandoc(docstring or "")
    head = _prefixed_head(doc) if doc else DOCSTRING_PREFIX + NO_DOCSTRING_SUMMARY
    if not doc:
        return head + "\n\n" + _new_google_notes(source)
    if is_numpy_style(doc):
        return _merge_numpy_notes(head, source)
    return _merge_google_notes(head, source)


def extract_scalar_source(doc: str) -> str:
    """Recover the scalar source embedded by :func:`build_docstring`.

    Takes the literal block following the first lead-in line (the
    marker), dedents it, and strips blank edges. Raises ``ValueError``
    when no marker or no block is present.
    """
    lines = doc.split("\n")
    lead_idx = next((i for i, line in enumerate(lines) if line.strip() == LEAD_IN), None)
    if lead_idx is None:
        raise ValueError("no array-vectorize source marker in the docstring")
    lead_indent = _indent_of(lines[lead_idx])
    block: list[str] = []
    for line in lines[lead_idx + 1 :]:
        if not line.strip():
            block.append("")
        elif _indent_of(line) > lead_indent:
            block.append(line)
        else:
            break
    while block and not block[0].strip():
        block.pop(0)
    while block and not block[-1].strip():
        block.pop()
    if not block:
        raise ValueError("no source block after the array-vectorize marker")
    margin = min(_indent_of(line) for line in block if line.strip())
    return "\n".join(line[margin:] if line.strip() else "" for line in block)
