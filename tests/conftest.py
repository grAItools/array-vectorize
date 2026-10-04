"""Shared pytest options."""

from __future__ import annotations

import os

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
