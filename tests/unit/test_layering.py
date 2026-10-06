"""Import-layering rules of the package, checked statically.

These encode the boundaries described in docs/architecture.md so a move
that breaks one fails here instead of in review.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

PACKAGE = "array_vectorize"
ROOT = Path(__file__).parents[2] / "src" / PACKAGE

#: runtime dependencies from pyproject.toml; anything else is dev-only
RUNTIME_DEPS = {"array_api_compat"}
#: the fuzzer is a dev CLI and may import dev dependencies (numpy)
DEV_ONLY_MODULES = {"fuzz"}


def _unit(path: Path) -> str:
    """Top-level unit of a source file: a subpackage or a top-level module."""
    parts = path.relative_to(ROOT).parts
    return parts[0] if len(parts) > 1 else path.stem


def _imports(path: Path) -> tuple[set[str], set[str]]:
    """(package units, external top-level modules) imported by ``path``."""
    package_dir = list(path.relative_to(ROOT).parts[:-1])
    internal: set[str] = set()
    external: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level:
            base = package_dir[: len(package_dir) - (node.level - 1)]
            target = base + (node.module.split(".") if node.module else [])
            if target:
                internal.add(target[0])
            else:
                internal.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            external.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            external.update(alias.name.split(".")[0] for alias in node.names)
    external.discard("__future__")
    return internal, external


SOURCES = sorted(ROOT.rglob("*.py"))


def _files_of(unit: str) -> list[Path]:
    return [path for path in SOURCES if _unit(path) == unit]


@pytest.mark.parametrize("path", _files_of("runtime"), ids=lambda p: p.name)
def test_runtime_is_a_stdlib_only_leaf(path: Path) -> None:
    # runtime helpers are injected into generated code and run on the
    # user's backend at call time; they must not drag in the compiler
    internal, external = _imports(path)
    assert internal <= {"runtime"}, internal
    assert external <= set(sys.stdlib_module_names), external


@pytest.mark.parametrize("path", _files_of("ir"), ids=lambda p: p.name)
def test_ir_depends_on_nothing_in_the_package(path: Path) -> None:
    internal, _ = _imports(path)
    assert internal <= {"ir"}, internal


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_only_the_package_root_imports_api(path: Path) -> None:
    # the pipeline reaches back into api only through the
    # helper_vectorizer callback, never by import
    internal, _ = _imports(path)
    if path != ROOT / "__init__.py":
        assert "api" not in internal


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_only_runtime_dependencies_are_imported(path: Path) -> None:
    if _unit(path) in DEV_ONLY_MODULES:
        return
    _, external = _imports(path)
    third_party = external - set(sys.stdlib_module_names) - {PACKAGE}
    assert third_party <= RUNTIME_DEPS, third_party
