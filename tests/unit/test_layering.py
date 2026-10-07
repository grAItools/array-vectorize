# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Import-layering rules of the package, checked statically.

These encode the boundaries described in docs/architecture.md so a move
that breaks one fails here instead of in review. The classifier is
AST-based, so it does not see dynamic imports (``importlib.import_module``,
``__import__``).
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

PACKAGE = "array_vectorize"
ROOT = pathlib.Path(__file__).parents[2] / "src" / PACKAGE
#: the unit of the package root ``__init__.py`` (its stem; a subpackage's
#: ``__init__.py`` instead belongs to that subpackage's unit)
ROOT_UNIT = "__init__"

#: runtime dependencies from pyproject.toml; anything else is dev-only
RUNTIME_DEPS = {"array_api_compat"}
#: the fuzzer is a dev CLI and may import dev dependencies (numpy)
DEV_ONLY_MODULES = {"fuzz"}


def _unit(path: pathlib.Path) -> str:
    """Top-level unit of a source file: a subpackage or a top-level module."""
    parts = path.relative_to(ROOT).parts
    return parts[0] if len(parts) > 1 else path.stem


def _is_unit(name: str) -> bool:
    """Whether ``name`` is a real module or subpackage under the package root."""
    return (ROOT / f"{name}.py").is_file() or (ROOT / name).is_dir()


def _classify(tree: ast.Module, package_dir: list[str]) -> tuple[set[str], set[str]]:
    """(package units, external top-level modules) imported by ``tree``.

    ``package_dir`` locates the importing file inside the package and
    resolves relative imports. Absolute self-imports are policed like
    relative ones: ``array_vectorize`` never counts as external.
    """
    internal: set[str] = set()
    external: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            base = package_dir[: len(package_dir) - (node.level - 1)]
            target = base + (node.module.split(".") if node.module else [])
            if target:
                internal.add(target[0])
            else:
                internal.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if parts[0] != PACKAGE:
                external.add(parts[0])
            elif len(parts) == 1:
                # `from array_vectorize import X, Y`: a real module or
                # subpackage X is its own unit; any other name is
                # re-exported by the root and reaches the root unit
                internal.update(
                    alias.name if _is_unit(alias.name) else ROOT_UNIT for alias in node.names
                )
            else:
                internal.add(parts[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == PACKAGE:
                    # importing the package, bare or dotted, reaches at
                    # least the root unit; the dotted form reaches the
                    # named unit
                    internal.add(parts[1] if len(parts) > 1 else ROOT_UNIT)
                else:
                    external.add(parts[0])
    external.discard("__future__")
    return internal, external


def _imports(path: pathlib.Path) -> tuple[set[str], set[str]]:
    """(package units, external top-level modules) imported by ``path``."""
    package_dir = list(path.relative_to(ROOT).parts[:-1])
    return _classify(ast.parse(path.read_text(encoding="utf-8")), package_dir)


SOURCES = sorted(ROOT.rglob("*.py"))


def _files_of(unit: str) -> list[pathlib.Path]:
    return [path for path in SOURCES if _unit(path) == unit]


@pytest.mark.parametrize(
    ("statement", "package_dir", "units"),
    [
        pytest.param("import array_vectorize", [], {ROOT_UNIT}, id="bare"),
        pytest.param("import array_vectorize as av", [], {ROOT_UNIT}, id="bare-aliased"),
        pytest.param("from array_vectorize import vectorize", [], {ROOT_UNIT}, id="re-export"),
        pytest.param("from array_vectorize import ir", [], {"ir"}, id="submodule"),
        pytest.param("from array_vectorize.ir import Node", [], {"ir"}, id="subpackage"),
        pytest.param("from array_vectorize.ir.nodes import Node", [], {"ir"}, id="nested"),
        pytest.param("import array_vectorize.pipeline", [], {"pipeline"}, id="dotted"),
        pytest.param("from . import errors", [], {"errors"}, id="relative-names"),
        pytest.param(
            "from .errors import VectorizationError", [], {"errors"}, id="relative-module"
        ),
        pytest.param(
            "from .validate import validate", ["frontend"], {"frontend"}, id="relative-inside"
        ),
        pytest.param(
            "from ..errors import VectorizationError",
            ["frontend"],
            {"errors"},
            id="relative-parent",
        ),
    ],
)
def test_import_statement_units(statement: str, package_dir: list[str], units: set[str]) -> None:
    """Every import form maps to the package units it reaches."""
    internal, external = _classify(ast.parse(statement), package_dir)
    assert internal == units
    assert PACKAGE not in external


@pytest.mark.parametrize("unit", ["runtime", "ir"])
def test_bare_package_import_breaks_the_leaf_rules(unit: str) -> None:
    # `import array_vectorize` classifies as the root unit, an internal
    # non-self unit, so the runtime-leaf and ir rules reject it
    internal, _ = _classify(ast.parse("import array_vectorize"), [unit])
    assert ROOT_UNIT in internal
    assert not internal <= {unit}


def test_known_real_imports() -> None:
    """The classifier sees the tree's real imports, not only synthetic ones."""
    pipeline_units, _ = _imports(ROOT / "pipeline.py")
    assert "lower" in pipeline_units
    fuzz_units, _ = _imports(ROOT / "fuzz.py")
    assert ROOT_UNIT in fuzz_units


@pytest.mark.parametrize("path", _files_of("runtime"), ids=lambda p: p.name)
def test_runtime_is_a_stdlib_only_leaf(path: pathlib.Path) -> None:
    # runtime helpers are injected into generated code and run on the
    # user's backend at call time; they must not drag in the compiler
    internal, external = _imports(path)
    assert internal <= {"runtime"}, internal
    assert external <= set(sys.stdlib_module_names), external


@pytest.mark.parametrize("path", _files_of("ir"), ids=lambda p: p.name)
def test_ir_depends_on_nothing_in_the_package(path: pathlib.Path) -> None:
    internal, _ = _imports(path)
    assert internal <= {"ir"}, internal


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_only_the_package_root_imports_api(path: pathlib.Path) -> None:
    # the pipeline reaches back into api only through the
    # helper_vectorizer callback, never by import; the root unit
    # re-exports api, so a dependency on it reaches api too
    internal, _ = _imports(path)
    if path == ROOT / "__init__.py":
        return
    assert "api" not in internal
    if path == ROOT / "fuzz.py":
        # the fuzz CLI is a dev tool and deliberately drives the
        # compiler through its public API
        return
    assert ROOT_UNIT not in internal


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_only_runtime_dependencies_are_imported(path: pathlib.Path) -> None:
    if _unit(path) in DEV_ONLY_MODULES:
        return
    _, external = _imports(path)
    third_party = external - set(sys.stdlib_module_names) - {PACKAGE}
    assert third_party <= RUNTIME_DEPS, third_party
