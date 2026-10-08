# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Tests for the mapping tables: every target must exist on array-api-strict."""

from __future__ import annotations

import math

import array_api_strict as xps

from array_vectorize.frontend import tables


def test_every_math_func_exists_on_strict() -> None:
    for xp_name, _, _ in tables.MATH_FUNCS.values():
        assert hasattr(xps, xp_name), f"xp.{xp_name} missing on array-api-strict"


def test_every_builtin_target_exists_on_strict() -> None:
    for xp_name in tables.BUILTIN_UNARY.values():
        assert hasattr(xps, xp_name)
    for xp_name in tables.BUILTIN_FOLDS.values():
        assert hasattr(xps, xp_name)
    for dtype in tables.BUILTIN_CASTS.values():
        assert hasattr(xps, dtype), f"xp.{dtype} missing on array-api-strict"
    for dtype in tables.MATH_SPECIAL.values():
        assert hasattr(xps, dtype)


def test_math_consts_match_math_module() -> None:
    for name, value in tables.MATH_CONSTS.items():
        if name == "nan":
            assert math.isnan(value)
        else:
            assert value == getattr(math, name)


def test_round_decimals_kwarg_is_not_standard() -> None:
    # round(x, n) is only supported if the standard supports it.
    # array-api-strict is the conformance reference; it must reject it.
    x = xps.asarray([1.23456])
    try:
        # Deliberately call an unsupported backend keyword to test its rejection.
        getattr(xps, "round")(x, decimals=2)  # noqa: B009
    except TypeError:
        pass
    else:
        raise AssertionError("array-api-strict accepts round(x, decimals=...)")


def test_hypot_is_binary_on_strict() -> None:
    # confirms the max_arity=2 entry for hypot
    a, b = xps.asarray([3.0]), xps.asarray([4.0])
    assert float(xps.hypot(a, b)[0]) == 5.0
