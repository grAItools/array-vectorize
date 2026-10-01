"""Shared pytest options."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("vectorizer")
    group.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Regenerate golden files for source snapshots",
    )
