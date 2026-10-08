# array-vectorize: compile scalar Python functions into exact Array API functions.
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
        # Semantics playground

        The hard part of scalar-to-array compilation is the numeric edges:
        booleans, mixed-width integers, `min`/`max` over mixed operands.
        Naive lowering either crashes on strict backends or silently
        rounds. `array-vectorize` routes these through runtime helpers that
        stay **exact whenever a single dtype can hold the result**.

        Every case below comes from the 26-round adversarial review that
        shaped these rules.
        """
    )
    return


@app.cell
def _():
    import numpy as np
    from array_vectorize import vectorize

    return np, vectorize


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Boolean arithmetic

        `True + True == 2` in Python, but strict backends reject boolean
        arithmetic outright and NumPy keeps booleans as `bool`. The
        helper computes the exact integer result.
        """
    )
    return


@app.cell
def _(np, vectorize):
    vec_bool_add = vectorize(lambda x: min(x, True) + True)
    vec_bool_pow = vectorize(lambda x: (x > 0) + 1)
    print("min(x, True) + True:", vec_bool_add(np.asarray([True, False])))
    print("(x > 0) + 1:        ", vec_bool_pow(np.asarray([2, -2])))
    return vec_bool_add, vec_bool_pow


@app.cell
def _(mo):
    mo.md(
        r"""
        ## uint64 beyond int64 range

        Values above `2**63 - 1` fit only uint64. Adding a non-negative
        int64 array stays exact in uint64; the remainder against the
        int64-min constant stays exact in int64.
        """
    )
    return


@app.cell
def _(np, vectorize):
    vec_u64_add = vectorize(lambda x, y: max(x, True) + y)
    x_huge = np.asarray([2**63 + 1], dtype=np.uint64)
    y_zero = np.asarray([0], dtype=np.int64)
    print("max(2**63+1, True) + 0 =", int(vec_u64_add(x_huge, y_zero)[0]))
    print("scalar Python          =", 2**63 + 1 + 0)
    return vec_u64_add, x_huge, y_zero


@app.cell
def _(np, vectorize):
    BOUND = -(2**63)

    vec_mod = vectorize(lambda x: max(x, True) % BOUND)
    print("max(1, True) % -(2**63) =", int(vec_mod(np.asarray([1], dtype=np.uint64))[0]))
    print("scalar Python           =", 1 % -(2**63))
    return BOUND, vec_mod


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Result-aware subtraction

        `y - max(x, True)` with huge `x` can produce results that only
        int64 can hold (negatives). The helper checks the actual values
        and picks the dtype that keeps every lane exact.
        """
    )
    return


@app.cell
def _(np, vectorize):
    vec_sub = vectorize(lambda x, y: y - max(x, True))
    print("int64(2**63-1) - max(uint64(2**63), True) =",
          int(vec_sub(np.asarray([2**63], dtype=np.uint64), np.asarray([2**63 - 1], dtype=np.int64))[0]))
    print("scalar Python                             =", (2**63 - 1) - 2**63)
    return (vec_sub,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## When nothing fits exactly

        A uint64-only positive in one lane and an int64-only negative in
        another fit no single dtype — the documented fallback is a
        float64 approximation (see the divergences table in the docs).
        """
    )
    return


@app.cell
def _(np, vec_sub):
    got = vec_sub(np.asarray([2**63 + 5, 1], dtype=np.uint64), np.asarray([3, 4], dtype=np.int64))
    print("mixed batch ->", got, "(float64 approximation, documented)")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Explore

        Edit the lambdas above and re-run — the playground is reactive.
        For the full rules see **Semantics & divergences** in the docs;
        for the rejection behavior try `while`, subscripts, or
        `round(x, 1)` and read the diagnostic.
        """
    )
    return


if __name__ == "__main__":
    app.run()
