# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

import marimo

__generated_with = "0.25.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Backends tour

        One generated function, every Array API backend. The generated
        code takes its namespace from the call arguments
        (`array_api_compat.array_namespace`), so the same `vec` runs on
        NumPy, `array-api-strict`, PyTorch, JAX, or CuPy arrays.
        """
    )
    return


@app.cell
def _():
    import math

    import numpy as np
    from array_vectorize import vectorize

    return math, np, vectorize


@app.cell
def _(math, vectorize):
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
    return clamp, psi


@app.cell
def _(np, psi):
    x_np = np.asarray([-2.0, -0.5, 0.0, 1.0, 3.0])
    psi(x_np)
    return (x_np,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## array-api-strict

        The strict reference backend enforces the standard aggressively:
        no boolean arithmetic, no implicit dtype promotion. The
        array-vectorize's runtime helpers handle those cases exactly — the
        same generated function works unchanged.
        """
    )
    return


@app.cell
def _(clamp, psi, x_np):
    import array_api_strict as xps

    sx = xps.asarray(x_np, dtype=xps.float64)
    print("psi:  ", psi(sx))
    print("clamp:", clamp(sx, hi=xps.asarray(0.5)))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Other backends

        With PyTorch, JAX, or CuPy installed, the same calls work on
        their arrays — try it:

        ```python
        import torch
        psi(torch.asarray(x_np))          # torch.Tensor
        ```

        No `vectorize`-time backend choice exists: the namespace is
        resolved per call from the arguments, so one compiled function
        serves every backend in the same process.
        """
    )
    return


if __name__ == "__main__":
    app.run()
