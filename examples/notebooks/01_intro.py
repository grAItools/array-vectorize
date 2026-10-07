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
        # array-vectorize — intro tour

        `vectorize(f)` compiles a **scalar** Python function into a
        semantically equivalent function over arrays, using only the
        Python Array API standard. The output is genuine vectorized code —
        not a Python-level element loop like `numpy.vectorize` — and the
        generated source is inspectable.

        Edit any cell; the notebook re-runs reactively.
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
    def sigmoid(x):
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        z = math.exp(x)
        return z / (1.0 + z)


    @vectorize
    def gelu(x):
        return 0.5 * x * (1.0 + math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


    @vectorize
    def l2_dist(x, y):
        d = 0.0
        for i in range(3):
            d = d + (x - y) * (x - y)
        return math.sqrt(d)
    return gelu, l2_dist, sigmoid


@app.cell
def _(gelu, l2_dist, np, sigmoid):
    x = np.asarray([-3.0, -1.0, 0.0, 1.0, 3.0])
    y = np.asarray([2.0, 0.5, -0.5, 0.5, 2.0])

    sigmoid(x), gelu(x), l2_dist(x, y)
    return x, y


@app.cell
def _(mo, sigmoid):
    mo.md(
        r"""
        ## The generated source

        `vec.source` shows exactly what runs — branches became `where`
        selects, the loop became a real loop with explicit phis. You can
        paste this into your own code or profile it.
        """
    )
    return


@app.cell
def _(sigmoid):
    print(sigmoid.source)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Is it actually faster?

        Compare the vectorized `sigmoid` against a scalar Python loop on
        1M values (best of 3, so scheduler noise does not dominate).
        """
    )
    return


@app.cell
def _(np, sigmoid):
    import time

    big = np.linspace(-6.0, 6.0, 1_000_000)


    def scalar_loop(arr):
        orig = sigmoid._vectorized_original
        return np.asarray([orig(float(v)) for v in arr])


    def best_of(fn, repeats=3):
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best


    t_vec = best_of(lambda: sigmoid(big))
    t_scalar = best_of(lambda: scalar_loop(big[:20_000])) * (1_000_000 / 20_000)
    print(f"vectorized: {t_vec * 1000:.1f} ms   scalar loop: ~{t_scalar * 1000:.0f} ms   speedup ~{t_scalar / t_vec:.0f}x")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## What is supported?

        A pragmatic scalar subset: arithmetic, comparisons, `and`/`or`/
        `not`, ternaries, `if/elif/else` with early returns, constant
        `range` loops, `min`/`max`/`abs`/`round`, casts, and most of
        `math`. Calls to other pure scalar functions are vectorized
        recursively. Anything else is **rejected with a precise
        diagnostic** — never silently miscompiled.

        Continue to the backends tour and the semantics playground.
        """
    )
    return


if __name__ == "__main__":
    app.run()
