# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Public API surface of the array_vectorize package."""

from __future__ import annotations

import numpy as np
import pytest

import array_vectorize


def test_package_exposes_version() -> None:
    assert array_vectorize.__version__


@pytest.mark.parametrize(
    "bad", [[1, 2], {"a": 1}, np.asarray([1.0])], ids=["list", "dict", "array"]
)
def test_unhashable_input_rejected_with_vectorization_error(bad: object) -> None:
    # the canonical cache lookup runs before extraction and must not
    # surface a TypeError for unhashable junk: the extractor's positioned
    # VectorizationError ("expected a plain Python function, got ...") wins
    with pytest.raises(array_vectorize.VectorizationError, match="plain Python function"):
        array_vectorize.vectorize(bad)  # type: ignore[arg-type]
