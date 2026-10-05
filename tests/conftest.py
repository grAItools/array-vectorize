"""Shared pytest options."""

from __future__ import annotations

import os
from importlib import import_module
from types import ModuleType

import pytest
from hypothesis import settings

settings.register_profile("ci", derandomize=True, max_examples=300, deadline=None)
settings.load_profile("ci" if os.environ.get("CI") else "default")


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("vectorizer")
    group.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Regenerate golden files for source snapshots",
    )


@pytest.fixture
def xps() -> ModuleType:
    """The array-api-strict backend (lazy: non-strict runs skip the import)."""
    return import_module("array_api_strict")


@pytest.fixture(scope="session")
def corpus() -> ModuleType:
    """The shared scalar-function corpus (tests/corpus.py, via pythonpath)."""
    return import_module("corpus")
