"""Targeted tests for remaining validator error branches."""

from __future__ import annotations

import importlib.util
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from vectorizer import vectorize
from vectorizer._errors import VectorizationError
from vectorizer._extract import extract_function
from vectorizer._validate import validate

_tmp = tempfile.TemporaryDirectory(prefix="vec_val2_")
_TMPDIR = Path(_tmp.name)
_seq = __import__("itertools").count()


def make_fn(body: str) -> Callable[..., Any]:
    path = _TMPDIR / f"snippet_{next(_seq)}.py"
    path.write_text("import math\n\n\ndef subject(x, y=2.0):\n" + body + "\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.subject


@pytest.mark.parametrize(
    ("body", "msg"),
    [
        ("    x //= 2\n    return x", None),  # control: valid augassign
        ("    x @= 2\n    return x", "augmented assignment"),
        ("    x = y = 1\n    return x", "multiple assignment targets"),
        ("    x.foo = 1\n    return x", "attribute assignment"),
        ("    del x\n    return x", None if False else "not supported"),
        ("    for i in range(3):\n        pass\n    else:\n        pass\n    return x", "for/else"),
        ("    return x + 9223372036854775808", "int64 range"),
        ("    return x", None),
    ],
)
def test_validator_branches(body: str, msg: str | None) -> None:
    if msg is None:
        validate(extract_function(make_fn(body)))
        return
    with pytest.raises(VectorizationError, match=msg):
        validate(extract_function(make_fn(body)))


def test_attribute_assignment() -> None:
    with pytest.raises(VectorizationError, match="attribute assignment"):
        validate(extract_function(make_fn("    a = x\n    a.f = 1\n    return a")))


def test_generic_attribute_access_rejected() -> None:
    with pytest.raises(VectorizationError, match="attribute access"):
        validate(extract_function(make_fn("    return x.real")))


def test_starred_call_argument_rejected() -> None:
    with pytest.raises(VectorizationError, match="starred call arguments"):
        validate(extract_function(make_fn("    return math.hypot(*[x, y])")))


def test_unknown_expression_falls_through() -> None:
    # Starred in a non-call context hits the generic expression message
    with pytest.raises(VectorizationError, match="not supported"):
        validate(extract_function(make_fn("    return x + (y := 1)")))


def test_del_is_rejected_by_vectorize() -> None:
    with pytest.raises(VectorizationError, match="statements are not supported"):
        vectorize(make_fn("    del x\n    return 1.0"))
