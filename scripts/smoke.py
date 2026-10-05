"""End-to-end smoke test (run directly)."""

import math

import numpy as np

from array_vectorize import vectorize


@vectorize
def psi(x):
    if x < 0:
        return 0.0
    return x * math.exp(-x)


@vectorize
def clamp(x, lo=0.0, hi=1.0):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def main() -> None:
    x = np.asarray([-2.0, -0.5, 0.0, 1.0, 3.0])
    expected = np.where(x < 0, 0.0, x * np.exp(-x))
    assert np.allclose(psi(x), expected), psi(x)

    expected_clamp = np.clip(x, 0.0, 1.0)
    assert np.allclose(clamp(x), expected_clamp), clamp(x)
    assert np.allclose(clamp(x, -1.0, 0.5), np.clip(x, -1.0, 0.5))

    print(psi.source)
    print(clamp.source)
    print("smoke ok")


if __name__ == "__main__":
    main()
