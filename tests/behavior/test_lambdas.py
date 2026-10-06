"""Vectorizing lambdas: identification, naming, values."""

from __future__ import annotations

import numpy as np

from array_vectorize import vectorize


def test_lambda_vectorize() -> None:
    vec = vectorize(lambda x: x * 2.0)
    got = vec(np.asarray([1.0, 2.0]))
    assert np.allclose(got, [2.0, 4.0])
    assert "def lambda_vec" in vec.source
