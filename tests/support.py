"""Shared helpers for the review-round regression tests."""

from __future__ import annotations

import importlib.util
import itertools
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vectorizer import vectorize

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
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.subject


def make_module(source: str) -> Any:
    """Write a standalone module from ``source`` and return the module."""
    path = TMPDIR / f"module_{next(_seq)}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def vec_of(body: str, extra: str = "") -> tuple[Callable[..., Any], Callable[..., Any]]:
    fn = make_fn(body, extra)
    return vectorize(fn), fn


def just_vec(body: str, extra: str = "") -> Callable[..., Any]:
    return vectorize(make_fn(body, extra))
