"""Public API surface of the array_vectorize package."""

from __future__ import annotations

import array_vectorize


def test_package_exposes_version() -> None:
    assert array_vectorize.__version__
