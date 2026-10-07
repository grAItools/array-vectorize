# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Shared test helpers (importable as ``support`` via pytest's pythonpath).

Black-box tests build their subjects from source snippets so each case is a
real module on disk (``inspect.getsource`` and lambda identification need
one): ``make_fn`` wraps a function body, ``make_module`` loads a whole
module, and ``vec_of`` / ``just_vec`` vectorize a snippet in one step.
``vfn`` vectorizes a function from the shared corpus (tests/corpus.py).
All snippet files live in ``TMPDIR``, removed at interpreter exit.
"""

from __future__ import annotations

from collections.abc import Callable
import importlib.util
import itertools
import pathlib
import tempfile
from typing import Any, cast, Protocol

import corpus

import array_vectorize
from array_vectorize import ir

_tmp = tempfile.TemporaryDirectory(prefix="vec_support_")
TMPDIR = pathlib.Path(_tmp.name)
_seq = itertools.count()


class Vectorized(Protocol):
    """Metadata attached dynamically to compiled functions, for test assertions."""

    source: str
    _vectorized_original: Callable[..., Any]
    __kwdefaults__: dict[str, Any] | None

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

    def with_namespace(self, xp: Any) -> Vectorized: ...


def with_metadata(fn: Callable[..., Any]) -> Vectorized:
    """Describe compiler-attached attributes without changing the public API.

    Tests asserting metadata use this boundary; ordinary calls still retain
    their callable types, and the assertions check the attributes at runtime.
    """
    return cast(Vectorized, fn)


def binding(stmt: ir.Stmt) -> ir.Binding:
    """Assert a statement is a binding before checking its name or expression."""
    assert isinstance(stmt, ir.Binding)
    return stmt


def backref(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Read the dynamically attached vectorization marker on a scalar function."""
    return cast(Callable[..., Any], getattr(fn, "__array_vectorized__"))  # noqa: B009


def make_fn(body: str, extra: str = "", defaults: str = "x, y=2.0") -> Callable[..., Any]:
    """Write a snippet module and return its ``subject`` function.

    ``extra`` is inserted into the module preamble (imports, module
    constants); ``defaults`` replaces the subject signature.
    """
    path = TMPDIR / f"snippet_{next(_seq)}.py"
    path.write_text(
        "import math\nimport numpy as np\n"
        + extra
        + "\n\n\ndef subject("
        + defaults
        + "):\n"
        + body
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return cast(Callable[..., Any], mod.subject)


def make_module(source: str) -> Any:
    """Write a standalone module from ``source`` and return the module."""
    path = TMPDIR / f"module_{next(_seq)}.py"
    path.write_text(source, encoding="utf-8", newline="\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def vec_of(body: str, extra: str = "") -> tuple[Callable[..., Any], Callable[..., Any]]:
    fn = make_fn(body, extra)
    return array_vectorize.vectorize(fn), fn


def just_vec(body: str, extra: str = "") -> Callable[..., Any]:
    return array_vectorize.vectorize(make_fn(body, extra))


def vfn(name: str) -> Callable[..., Any]:
    """Vectorize the corpus function ``name`` (tests/corpus.py)."""
    return array_vectorize.vectorize(getattr(corpus, name))
