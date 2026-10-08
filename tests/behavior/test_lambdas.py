# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Vectorizing lambdas: identification, naming, values."""

from __future__ import annotations

import math

import numpy as np
import support

import array_vectorize


def test_lambda_vectorize() -> None:
    vec = array_vectorize.vectorize(lambda x: x * 2.0)
    got = vec(np.asarray([1.0, 2.0]))
    assert np.allclose(got, [2.0, 4.0])
    assert "def lambda_vec" in support.with_metadata(vec).source


# ------------------------------------------------------- lambda identification


def test_multiple_lambdas_on_one_line() -> None:
    mod = support.make_module("f1, f2 = (lambda x: x + 1), (lambda x: x + 2)\n")
    assert array_vectorize.vectorize(mod.f2)(np.asarray([1.0]))[0] == 3.0
    assert array_vectorize.vectorize(mod.f1)(np.asarray([1.0]))[0] == 2.0


def test_lambdas_differing_only_in_names() -> None:
    mod = support.make_module(
        "import math\nf1, f2 = (lambda x: math.sin(x)), (lambda x: math.cos(x))\n"
    )
    assert np.isclose(array_vectorize.vectorize(mod.f2)(np.asarray([1.0]))[0], math.cos(1.0))
    assert np.isclose(array_vectorize.vectorize(mod.f1)(np.asarray([1.0]))[0], math.sin(1.0))


def test_lambdas_differing_only_in_defaults() -> None:
    mod = support.make_module("g1, g2 = (lambda x, y=1: x + y), (lambda x, y=2: x + y)\n")
    assert array_vectorize.vectorize(mod.g2)(np.asarray([1.0]))[0] == 3.0
    assert array_vectorize.vectorize(mod.g1)(np.asarray([1.0]))[0] == 2.0


def test_lambda_signed_zero_constants() -> None:
    mod = support.make_module(
        "import math\n"
        "f1, f2 = (\n"
        "    lambda x: math.copysign(x, 0.0),\n"
        "    lambda x: math.copysign(x, -0.0),\n"
        ")\n"
    )
    assert array_vectorize.vectorize(mod.f2)(np.asarray([1.0]))[0] == -1.0
    assert array_vectorize.vectorize(mod.f1)(np.asarray([1.0]))[0] == 1.0


def test_lambda_int_vs_bool_vs_float_constants() -> None:
    mod = support.make_module(
        "g1, g2, g3 = (lambda x: x + 1), (lambda x: x + True), (lambda x: x + 1.0)\n"
    )
    assert array_vectorize.vectorize(mod.g2)(np.asarray([1.0]))[0] == 2.0
    assert array_vectorize.vectorize(mod.g3)(np.asarray([1.0]))[0] == 2.0
