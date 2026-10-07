# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Generated-name allocation: reserved params, shadowed helper names."""

from __future__ import annotations

import math

import numpy as np
import support

import array_vectorize

X = np.asarray([-2.0, -0.5, 0.0, 0.5, 2.0, 10.0])


# ------------------------------------------------------------- name allocation


def test_reserved_param() -> None:
    assert np.allclose(support.vfn("reserved_param")(X), X + 1)


def test_reserved_xp_param_keyword_call() -> None:
    mod = support.make_module("import numpy as np\n\n\ndef subject(xp):\n    return xp + 1\n")
    vec = array_vectorize.vectorize(mod.subject)
    assert np.allclose(vec(xp=np.asarray([1.0])), [2.0])
    assert "def subject_vec(xp):" in support.with_metadata(vec).source


def test_namespace_var_rebinding_no_collision() -> None:
    mod = support.make_module(
        "import math\n\n\ndef subject(xp):\n    xp = xp + 1\n    return math.sin(xp)\n"
    )
    vec = array_vectorize.vectorize(mod.subject)
    assert np.allclose(vec(xp=np.asarray([1.0])), [math.sin(2.0)])


def test_reserved_caller_name_param_not_renamed() -> None:
    mod = support.make_module(
        "def reserved_caller(reserved_caller_vec):\n    return reserved_caller_vec + 1\n"
    )
    vec = array_vectorize.vectorize(mod.reserved_caller)
    assert np.allclose(vec(reserved_caller_vec=np.asarray([1.0])), [2.0])


def test_helper_name_shadowed_by_param() -> None:
    vec = array_vectorize.vectorize(
        support.make_fn("    return min(_vec_common_dtype, 3)", defaults="_vec_common_dtype")
    )
    got = vec(np.asarray([1.0]))
    assert got[0] == 1.0


def test_helper_names_shadowed_by_params() -> None:
    vec = array_vectorize.vectorize(
        support.make_fn(
            "    return min(_vec_arith_dtype, _vec_common_dtype) + (_vec_arith_dtype > 0)",
            defaults="_vec_arith_dtype, _vec_common_dtype=2.0",
        )
    )
    got = vec(np.asarray([1.0]))
    assert got[0] == 2.0  # min(1, 2) + True
