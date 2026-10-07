# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Source inspection and repository text I/O across platforms."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import pathlib

import numpy as np
import pytest
import support

import array_vectorize

ROOT = pathlib.Path(__file__).parents[1]


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_unicode_source_and_path_are_inspectable_and_vectorizable(
    tmp_path: pathlib.Path, newline: str
) -> None:
    directory = tmp_path / "données 日本語"
    directory.mkdir()
    path = directory / "résumé.py"
    source = 'def subject(x):\n    """Résumé: 日本語."""\n    return x + 2.0\n'
    path.write_bytes(source.replace("\n", newline).encode("utf-8"))
    spec = importlib.util.spec_from_file_location("portability_subject", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert inspect.getsource(module.subject) == source
    vec = array_vectorize.vectorize(module.subject)
    values = np.asarray([-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(vec(values), [module.subject(float(value)) for value in values])
    assert "Résumé: 日本語." in support.with_metadata(vec).source


def test_snippet_helpers_write_utf8_and_lf(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(support, "TMPDIR", tmp_path)
    support.make_fn('    """Résumé: 日本語."""\n    return x + y')
    support.make_module('def subject(x):\n    """Résumé: 日本語."""\n    return x\n')
    paths = list(tmp_path.glob("*.py"))
    assert len(paths) == 2
    for path in paths:
        raw = path.read_bytes()
        assert "Résumé: 日本語." in raw.decode("utf-8")
        assert b"\r" not in raw


def test_repository_path_text_operations_have_explicit_utf8() -> None:
    failures = []
    for directory in ("src", "tests", "scripts"):
        for path in (ROOT / directory).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"read_text", "write_text"}
                ):
                    encoding = next(
                        (keyword.value for keyword in node.keywords if keyword.arg == "encoding"),
                        None,
                    )
                    if not isinstance(encoding, ast.Constant) or encoding.value != "utf-8":
                        failures.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not failures, f"Path text operations need explicit UTF-8: {failures}"
