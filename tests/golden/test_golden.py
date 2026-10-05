"""Golden source snapshots (plan T2): ast.dump equality against tests/golden/cases."""

from __future__ import annotations

import ast
from pathlib import Path

import corpus
import pytest
from corpus import GOLDEN_NAMES

from vectorizer import vectorize

GOLDEN_DIR = Path(__file__).parent / "cases"


def _generated_source(name: str) -> str:
    fn = getattr(corpus, name)
    return vectorize(fn).source


def _golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.py"


def _tree(source: str) -> str:
    return ast.dump(ast.parse(source))


@pytest.mark.parametrize("name", GOLDEN_NAMES)
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


def test_golden_files_are_valid_python() -> None:
    for path in GOLDEN_DIR.glob("*.py"):
        ast.parse(path.read_text())
