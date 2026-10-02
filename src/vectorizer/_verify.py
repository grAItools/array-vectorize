"""Differential-verification helper (plan §3, M4).

``vectorize(f, verify=example_args)`` runs the generated function and the
original scalar function on the example inputs at generation time and raises
:class:`VectorizationError` if they disagree element-wise (NaN-aware, small
tolerance). Uses only Array API functions, so it works on any backend.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

from array_api_compat import array_namespace

from ._errors import VectorizationError

__all__ = ["verify_match"]

_RTOL = 1e-9
_ATOL = 1e-9


def _is_array(value: Any) -> bool:
    return hasattr(value, "__array_namespace__")


def _py_scalar(value: Any) -> Any:
    """Backend scalar -> plain Python scalar (preserves int/bool/float type)."""
    item = getattr(value, "item", None)
    return item() if item is not None else value


def verify_match(
    vec: Callable[..., Any],
    original: Callable[..., Any],
    example_args: tuple[Any, ...],
) -> None:
    """Check ``vec(*example_args)`` against element-wise ``original`` calls."""
    positions = [i for i, a in enumerate(example_args) if _is_array(a)]
    if not positions:
        raise VectorizationError("verify= needs at least one Array API array in the example inputs")
    arrays = [example_args[i] for i in positions]
    xp = array_namespace(*arrays)
    with warnings.catch_warnings():
        # out-of-domain lanes are expected (plan D3); their warnings are noise
        # (including inf-inf in the comparison below)
        warnings.simplefilter("ignore", RuntimeWarning)
        got = vec(*example_args)
        broadcast = xp.broadcast_arrays(*arrays)
        shape = broadcast[0].shape
        if tuple(got.shape) != tuple(shape):
            raise VectorizationError(
                f"verification failed: result shape {tuple(got.shape)} "
                f"!= input shape {tuple(shape)}"
            )
        nan = float("nan")
        flat = [xp.reshape(a, (-1,)) for a in broadcast]
        expected_values: list[Any] = []
        for elems in zip(*flat, strict=True):
            # pass through non-array example arguments positionally, and give the
            # oracle plain Python scalars (backend scalars would change e.g.
            # int/bool arithmetic semantics)
            call_args = list(example_args)
            for pos, elem in zip(positions, elems, strict=True):
                call_args[pos] = _py_scalar(elem)
            try:
                expected_values.append(original(*call_args))
            except (ValueError, ArithmeticError, OverflowError):
                # plan D3: scalar exceptions become IEEE values (NaN/inf) when
                # vectorized, so a raising lane expects NaN here
                expected_values.append(nan)
        expected = xp.asarray(expected_values)
        got_flat = xp.reshape(got, (-1,))
        # compare values with their natural types (bools as 0/1) via float64;
        # coercing the expected side to the output dtype could hide miscompiles
        got_f = xp.astype(got_flat, xp.float64)
        exp_f = xp.astype(xp.reshape(xp.asarray(expected_values), (-1,)), xp.float64)
        # exact equality first (covers matching infinities and identical values);
        # the tolerance term is gated on finiteness so inf-vs-finite mismatches
        # cannot slip through as inf <= inf
        diff = xp.abs(got_f - exp_f)
        close = (got_f == exp_f) | (xp.isfinite(diff) & (diff <= _ATOL + _RTOL * xp.abs(exp_f)))
        both_nan = xp.isnan(got_f) & xp.isnan(exp_f)
        if not bool(xp.all(close | both_nan)):
            raise VectorizationError(
                "verification failed: vectorized output differs from the scalar "
                f"original on the provided example inputs (got {got}, expected {expected})"
            )
