"""Golden source snapshots: ast.dump equality against tests/golden/cases."""

from __future__ import annotations

import ast
import pathlib

import corpus
import numpy as np
import pytest
import support

import array_vectorize

GOLDEN_DIR = pathlib.Path(__file__).parent / "cases"

#: pinned-variant goldens (vectorize(namespace=np)): the namespace object
#: never appears in the source text, only the ``xp = _namespace`` binding
PINNED_GOLDEN_NAMES = ["add"]


def _generated_source(name: str) -> str:
    fn = getattr(corpus, name)
    return support.with_metadata(array_vectorize.vectorize(fn)).source


def _generated_pinned_source(name: str) -> str:
    fn = getattr(corpus, name)
    return support.with_metadata(array_vectorize.vectorize(fn, namespace=np)).source


def _golden_path(name: str) -> pathlib.Path:
    return GOLDEN_DIR / f"{name}.py"


def _pinned_golden_path(name: str) -> pathlib.Path:
    return GOLDEN_DIR / f"{name}_pinned.py"


def _tree(source: str) -> str:
    return ast.dump(ast.parse(source))


@pytest.mark.parametrize("name", corpus.GOLDEN_NAMES)
def test_golden_source(name: str, request: pytest.FixtureRequest) -> None:
    source = _generated_source(name)
    path = _golden_path(name)
    if request.config.getoption("--update-golden"):
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        return
    assert path.exists(), f"missing golden file {path}; run pytest --update-golden"
    golden = path.read_text()
    assert _tree(source) == _tree(golden), (
        f"generated source differs from golden for {name!r}; "
        "review the diff and regenerate with pytest --update-golden if intended\n"
        f"--- generated ---\n{source}"
    )


@pytest.mark.parametrize("name", PINNED_GOLDEN_NAMES)
def test_golden_pinned_source(name: str, request: pytest.FixtureRequest) -> None:
    source = _generated_pinned_source(name)
    path = _pinned_golden_path(name)
    if request.config.getoption("--update-golden"):
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        return
    assert path.exists(), f"missing golden file {path}; run pytest --update-golden"
    golden = path.read_text()
    assert _tree(source) == _tree(golden), (
        f"generated pinned source differs from golden for {name!r}; "
        "review the diff and regenerate with pytest --update-golden if intended\n"
        f"--- generated ---\n{source}"
    )


def test_golden_files_are_valid_python() -> None:
    for path in GOLDEN_DIR.glob("*.py"):
        ast.parse(path.read_text())
