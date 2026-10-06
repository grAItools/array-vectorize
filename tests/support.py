"""Shared test helpers (importable as ``support`` via pytest's pythonpath).

Black-box tests build their subjects from source snippets so each case is a
real module on disk (``inspect.getsource`` and lambda identification need
one): ``make_fn`` wraps a function body, ``make_module`` loads a whole
module, and ``vec_of`` / ``just_vec`` vectorize a snippet in one step.
``vfn`` vectorizes a function from the shared corpus (tests/corpus.py).
All snippet files live in ``TMPDIR``, removed at interpreter exit.
"""

from __future__ import annotations

import importlib.util
import itertools
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import corpus

from array_vectorize import vectorize

_tmp = tempfile.TemporaryDirectory(prefix="vec_support_")
TMPDIR = Path(_tmp.name)
_seq = itertools.count()


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
        + "\n"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.subject


def make_module(source: str) -> Any:
    """Write a standalone module from ``source`` and return the module."""
    path = TMPDIR / f"module_{next(_seq)}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def vec_of(body: str, extra: str = "") -> tuple[Callable[..., Any], Callable[..., Any]]:
    fn = make_fn(body, extra)
    return vectorize(fn), fn


def just_vec(body: str, extra: str = "") -> Callable[..., Any]:
    return vectorize(make_fn(body, extra))


def vfn(name: str) -> Callable[..., Any]:
    """Vectorize the corpus function ``name`` (tests/corpus.py)."""
    return vectorize(getattr(corpus, name))
