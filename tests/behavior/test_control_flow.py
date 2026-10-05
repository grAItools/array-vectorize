"""Branch merging, early returns, ternaries, piecewise functions, SSA rebinds."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from support import make_fn

from array_vectorize import vectorize

spec = importlib.util.spec_from_file_location(
    "vec_corpus_b", Path(__file__).parent.parent / "corpus.py"
)
assert spec is not None and spec.loader is not None
CORPUS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CORPUS)


def vfn(name: str) -> Any:
    return vectorize(getattr(CORPUS, name))


X = np.asarray([-2.0, -0.5, 0.0, 0.5, 2.0, 10.0])
Y = np.asarray([3.0, 1.5, 1.0, 0.5, -1.0, -3.0])


# ---- from test_behavior (git history: tests/test_behavior.py)


def test_ternary() -> None:
    expected = np.where(X > 0, X * 2, X / 2)
    assert np.allclose(vfn("ternary")(X), expected)


def test_ssa_rebind() -> None:
    expected = (X + 1) * 2 + 3
    assert np.allclose(vfn("ssa_rebind")(X), expected)


@pytest.mark.parametrize("name", ["relu", "relu_with_else"])
def test_relu_variants(name: str) -> None:
    assert np.allclose(vfn(name)(X), np.maximum(X, 0.0))


def test_psi() -> None:
    expected = np.where(X < 0, 0.0, X * np.exp(-X))
    assert np.allclose(vfn("psi")(X), expected)


def test_piecewise() -> None:
    expected = np.where(X < -1, -1.0, np.where(X > 1, 1.0, X))
    assert np.allclose(vfn("piecewise")(X), expected)


def test_merge_with_prior() -> None:
    assert np.allclose(vfn("merge_with_prior")(X), np.where(X > 0, X, 0.0))


def test_nested_early_returns() -> None:
    expected = np.where(X > 0, np.where(X > 10, 1.0, 2.0), 3.0)
    assert np.allclose(vfn("nested_early_returns")(X), expected)


def test_both_branches_assign() -> None:
    expected = np.where(X > 0, X, -X) + np.where(X > 0, 1.0, 2.0)
    assert np.allclose(vfn("both_branches_assign")(X), expected)


# ---- from test_review_round14 (git history: tests/test_review_round14.py)


def test_branch_merged_maybe_bool_arithmetic() -> None:
    vec = vectorize(
        make_fn(
            "    if x == True:\n"
            "        a = min(x, True)\n"
            "    else:\n"
            "        a = max(x, False)\n"
            "    return a + a"
        )
    )
    assert list(np.asarray(vec(np.asarray([True, False])))) == [2, 0]
