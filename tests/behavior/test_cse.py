# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""CSE temps: opportunities, SSA binding collisions, signed zeros."""

from __future__ import annotations

import numpy as np
import support

X = np.asarray([-2.0, -0.5, 0.0, 0.5, 2.0, 10.0])


# ------------------------------------------------------------------- CSE temps


def test_cse_opportunity() -> None:
    s = np.sqrt(np.abs(X))
    expected = s * s + s
    assert np.allclose(support.vfn("cse_opportunity")(np.abs(X)), expected)


def test_cse_temp_does_not_collide_with_ssa_bindings() -> None:
    vec, fn = support.vec_of(
        "    t = x + 1\n    t = t + 1\n    return t + math.sin(x) + math.sin(x)"
    )
    xs = np.asarray([1.0])
    assert np.isclose(vec(xs)[0], fn(1.0))


def test_signed_zero_not_cse_merged() -> None:
    vec = support.just_vec("    return math.copysign(x, 0.0) + math.copysign(x, -0.0)")
    assert np.allclose(vec(np.asarray([1.0])), [0.0])
