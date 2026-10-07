# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Branch merging, early returns, ternaries, piecewise functions, SSA rebinds."""

from __future__ import annotations

import numpy as np
import pytest
import support

import array_vectorize

X = np.asarray([-2.0, -0.5, 0.0, 0.5, 2.0, 10.0])


# -------------------------------------------------- branches and early returns


def test_ternary() -> None:
    expected = np.where(X > 0, X * 2, X / 2)
    assert np.allclose(support.vfn("ternary")(X), expected)


def test_ssa_rebind() -> None:
    expected = (X + 1) * 2 + 3
    assert np.allclose(support.vfn("ssa_rebind")(X), expected)


@pytest.mark.parametrize("name", ["relu", "relu_with_else"])
def test_relu_variants(name: str) -> None:
    assert np.allclose(support.vfn(name)(X), np.maximum(X, 0.0))


def test_psi() -> None:
    expected = np.where(X < 0, 0.0, X * np.exp(-X))
    assert np.allclose(support.vfn("psi")(X), expected)


def test_piecewise() -> None:
    expected = np.where(X < -1, -1.0, np.where(X > 1, 1.0, X))
    assert np.allclose(support.vfn("piecewise")(X), expected)


def test_merge_with_prior() -> None:
    assert np.allclose(support.vfn("merge_with_prior")(X), np.where(X > 0, X, 0.0))


def test_nested_early_returns() -> None:
    expected = np.where(X > 0, np.where(X > 10, 1.0, 2.0), 3.0)
    assert np.allclose(support.vfn("nested_early_returns")(X), expected)


def test_both_branches_assign() -> None:
    expected = np.where(X > 0, X, -X) + np.where(X > 0, 1.0, 2.0)
    assert np.allclose(support.vfn("both_branches_assign")(X), expected)


# ----------------------------------------------- branch merging of value kinds


def test_branch_merged_maybe_bool_arithmetic() -> None:
    vec = array_vectorize.vectorize(
        support.make_fn(
            "    if x == True:\n"
            "        a = min(x, True)\n"
            "    else:\n"
            "        a = max(x, False)\n"
            "    return a + a"
        )
    )
    assert list(np.asarray(vec(np.asarray([True, False])))) == [2, 0]


# --------------------------------------------------------------- boolop chains


def test_deeply_nested_boolop_chain() -> None:
    vec = array_vectorize.vectorize(support.make_fn("    return x and y and x and y"))
    import numpy as np

    xs = np.asarray([0.0, 1.0, 2.0])
    ys = np.asarray([3.0, 0.0, 5.0])
    out = vec(xs, ys)
    assert np.allclose(out, [0.0, 0.0, 5.0])
