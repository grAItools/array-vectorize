"""Public API surface of the array_vectorize package."""

from __future__ import annotations

import numpy as np
import pytest

import array_vectorize
from array_vectorize import VectorizationError, vectorize


def test_package_exposes_version() -> None:
    assert array_vectorize.__version__


@pytest.mark.parametrize(
    "bad", [[1, 2], {"a": 1}, np.asarray([1.0])], ids=["list", "dict", "array"]
)
def test_unhashable_input_rejected_with_vectorization_error(bad: object) -> None:
    # the canonical cache lookup runs before extraction and must not
    # surface a TypeError for unhashable junk: the extractor's positioned
    # VectorizationError ("expected a plain Python function, got ...") wins
    with pytest.raises(VectorizationError, match="plain Python function"):
        vectorize(bad)  # type: ignore[arg-type]
