# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""End-to-end smoke test (run directly)."""

import math

import numpy as np

import array_vectorize


@array_vectorize.vectorize
def psi(x: float) -> float:
    """Exponential decay on the positive half, zero below it."""
    if x < 0:
        return 0.0
    return x * math.exp(-x)


@array_vectorize.vectorize
def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp x into [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def main() -> None:
    """Run the smoke checks and print the generated sources."""
    x = np.asarray([-2.0, -0.5, 0.0, 1.0, 3.0])
    expected = np.where(x < 0, 0.0, x * np.exp(-x))
    assert np.allclose(psi(x), expected), psi(x)

    expected_clamp = np.clip(x, 0.0, 1.0)
    assert np.allclose(clamp(x), expected_clamp), clamp(x)
    assert np.allclose(clamp(x, -1.0, 0.5), np.clip(x, -1.0, 0.5))

    print(array_vectorize.get_source(psi))
    print(array_vectorize.get_source(clamp))
    print("smoke ok")


if __name__ == "__main__":
    main()
