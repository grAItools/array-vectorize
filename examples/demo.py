"""Runnable example: vectorize a few scalar functions and compare backends."""

import math

import numpy as np

from array_vectorize import vectorize


@vectorize
def sigmoid(x):
    """Numerically stable logistic sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


@vectorize
def gelu(x):
    """GELU activation (tanh approximation)."""
    return 0.5 * x * (1.0 + math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


@vectorize
def l2_dist(x, y):
    """Distance accumulated in a constant-trip loop, then sqrt."""
    d = 0.0
    for i in range(3):
        d = d + (x - y) * (x - y)
    return math.sqrt(d)


def main() -> None:
    """Run the demo on NumPy and array-api-strict inputs."""
    x = np.asarray([-3.0, -1.0, 0.0, 1.0, 3.0])
    y = np.asarray([2.0, 0.5, -0.5, 0.5, 2.0])

    print("sigmoid:", sigmoid(x))
    print("gelu:   ", gelu(x))
    print("l2:     ", l2_dist(x, y))

    # array-api-strict backend, same generated functions
    import array_api_strict as xps

    sx = xps.asarray([-3.0, -1.0, 0.0, 1.0, 3.0])
    print("strict: ", list(map(float, sigmoid(sx))))

    # generated source is inspectable
    print("\n--- gelu.source ---")
    print(gelu.source)


if __name__ == "__main__":
    main()
