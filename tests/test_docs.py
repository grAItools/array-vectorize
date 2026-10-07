# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Check public documentation against its executable sources of truth."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import inspect
import pathlib
import re
import subprocess
import sys
import tomllib

import pytest

import array_vectorize
from array_vectorize.frontend import tables

ROOT = pathlib.Path(__file__).parents[1]


def _read_doc(name: str) -> str:
    return (ROOT / "docs" / name).read_text(encoding="utf-8")


def _assert_identifiers_documented(document: str, identifiers: Iterable[str]) -> None:
    """Require complete identifiers inside Markdown code spans."""
    documented = {
        identifier
        for span in re.findall(r"(?<!`)`([^`\n]+)`(?!`)", document)
        for identifier in re.findall(r"(?<![\w.])[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", span)
    }
    missing = set(identifiers) - documented
    assert not missing, f"Undocumented identifiers: {sorted(missing)}"


def _assert_defaults_documented(document: str, defaults: Mapping[str, str]) -> None:
    """Compare the parameter table with runtime defaults, without pinning prose."""
    documented = dict(
        re.findall(r"^\| `([a-z_]\w*)` \| `([^`]+)` \|", document, flags=re.MULTILINE)
    )
    actual = {name: documented.get(name) for name in defaults}
    assert actual == dict(defaults), f"Documented defaults {actual} differ from {dict(defaults)}"


def _assert_options_documented(document: str, help_text: str) -> None:
    """Require every long CLI option to have an explicit code span."""
    options = set(re.findall(r"(?<![\w-])--[a-z][a-z-]*", help_text))
    documented = set(re.findall(r"`(--[a-z][a-z-]*)`", document))
    missing = options - documented
    assert not missing, f"Undocumented CLI options: {sorted(missing)}"


def test_public_exports_are_documented() -> None:
    _assert_identifiers_documented(_read_doc("api.md"), array_vectorize.__all__)


def test_vectorize_keyword_defaults_are_documented() -> None:
    defaults = {
        name: repr(parameter.default)
        for name, parameter in inspect.signature(array_vectorize.vectorize).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    _assert_defaults_documented(_read_doc("api.md"), defaults)


def test_supported_mapping_identifiers_are_documented() -> None:
    math_names = set(tables.MATH_FUNCS) | set(tables.MATH_SPECIAL) | set(tables.MATH_CONSTS)
    builtin_names = (
        set(tables.BUILTIN_UNARY) | set(tables.BUILTIN_FOLDS) | set(tables.BUILTIN_CASTS)
    )
    _assert_identifiers_documented(
        _read_doc("supported-subset.md"), {f"math.{name}" for name in math_names} | builtin_names
    )


def test_fuzzer_options_are_documented() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "array_vectorize.fuzz", "--help"],
        check=True,
        capture_output=True,
        encoding="utf-8",
        cwd=ROOT,
    )
    _assert_options_documented(_read_doc("api.md"), result.stdout)


def test_python_requirement_is_documented() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirement = metadata["project"]["requires-python"]
    documented = re.search(r"Requires Python\s+([^\n.]+(?:\.\d+)*)\.", _read_doc("index.md"))
    assert documented is not None, "Missing Python requirement"
    assert documented[1].replace(" ", "") == requirement


@pytest.mark.parametrize("name", ["get_source", "math.sin", "int"])
def test_identifier_check_rejects_missing_and_partial_names(name: str) -> None:
    with pytest.raises(AssertionError, match="Undocumented identifiers"):
        _assert_identifiers_documented(f"`{name}_extra` and prose {name}", [name])


@pytest.mark.parametrize("document", ["", "| `strict` | `False` | changed default |"])
def test_default_check_rejects_missing_and_changed_defaults(document: str) -> None:
    with pytest.raises(AssertionError, match="Documented defaults"):
        _assert_defaults_documented(document, {"strict": "True"})


def test_cli_check_rejects_missing_and_partial_options() -> None:
    with pytest.raises(AssertionError, match="Undocumented CLI options"):
        _assert_options_documented("`--seed-extra`", "usage: tool --seed SEED")
