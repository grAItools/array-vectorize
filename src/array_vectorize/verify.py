"""Differential-verification helper.

``vectorize(f, verify=example_args)`` runs the generated function and the
original scalar function on the example inputs at generation time and raises
:class:`VectorizationError` if they disagree element-wise (NaN-aware, small
tolerance). Uses only Array API functions, so it works on any backend.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import warnings

import array_api_compat

from array_vectorize import compat
from array_vectorize import errors

__all__ = ["verify_match"]

_RTOL = 1e-9
_ATOL = 1e-9


def verify_match(
    vec: Callable[..., Any],
    original: Callable[..., Any],
    example_args: tuple[Any, ...],
    namespace: Any = None,
) -> None:
    """Check ``vec(*example_args)`` against element-wise ``original`` calls.

    With ``namespace`` set (a pinned vectorization), that namespace is used
    directly instead of extracting it from the example arrays.
    """
    positions = [i for i, a in enumerate(example_args) if compat._is_array(a)]
    if not positions:
        raise errors.VectorizationError(
            "verify= needs at least one Array API array in the example inputs"
        )
    arrays = [example_args[i] for i in positions]
    xp = namespace if namespace is not None else array_api_compat.array_namespace(*arrays)
    with warnings.catch_warnings():
        # out-of-domain lanes are expected (design D3); their warnings are noise
        # (including inf-inf in the comparison below)
        warnings.simplefilter("ignore", RuntimeWarning)
        got = vec(*example_args)
        broadcast = xp.broadcast_arrays(*arrays)
        shape = broadcast[0].shape
        # constant-return functions produce a Python scalar or 0-d array;
        # broadcast it to the input shape so element-wise comparison works.
        # asarray on a plain Python scalar uses the backend's default dtype
        # (torch: float32) — build it in float64 so precision survives to
        # the comparison below
        got_arr = got if hasattr(got, "shape") else xp.asarray(got, dtype=xp.float64)
        if got_arr.shape == ():
            got_arr = xp.broadcast_to(got_arr, shape)
        if tuple(got_arr.shape) != tuple(shape):
            raise errors.VectorizationError(
                f"verification failed: result shape {tuple(got_arr.shape)} "
                f"!= input shape {tuple(shape)}"
            )
        got = got_arr
        nan = float("nan")
        flat = [xp.reshape(a, (-1,)) for a in broadcast]
        expected_values: list[Any] = []
        for elems in zip(*flat, strict=True):
            # pass through non-array example arguments positionally, and give the
            # oracle plain Python scalars (backend scalars would change e.g.
            # int/bool arithmetic semantics)
            call_args = list(example_args)
            for pos, elem in zip(positions, elems, strict=True):
                call_args[pos] = compat._py_scalar(elem)
            try:
                expected_values.append(original(*call_args))
            except (ValueError, ArithmeticError, OverflowError):
                # design D3: scalar exceptions become IEEE values (NaN/inf) when
                # vectorized, so a raising lane expects NaN here
                expected_values.append(nan)
        expected = xp.asarray(expected_values)
        got_flat = xp.reshape(got, (-1,))
        # compare values with their natural types (bools as 0/1) via float64;
        # coercing the expected side to the output dtype could hide miscompiles
        got_f = xp.astype(got_flat, xp.float64)
        # asarray on a list of Python scalars uses the backend's default
        # dtype (torch: float32), which would lose precision BEFORE the
        # astype below — build the expected side in float64 directly
        exp_f = xp.reshape(xp.asarray(expected_values, dtype=xp.float64), (-1,))
        # exact equality first (covers matching infinities and identical values);
        # the tolerance term is gated on finiteness so inf-vs-finite mismatches
        # cannot slip through as inf <= inf
        diff = xp.abs(got_f - exp_f)
        close = (got_f == exp_f) | (xp.isfinite(diff) & (diff <= _ATOL + _RTOL * xp.abs(exp_f)))
        both_nan = xp.isnan(got_f) & xp.isnan(exp_f)
        if not bool(xp.all(close | both_nan)):
            raise errors.VectorizationError(
                "verification failed: vectorized output differs from the scalar "
                f"original on the provided example inputs (got {got}, expected {expected})"
            )
