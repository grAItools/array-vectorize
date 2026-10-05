"""Opt-in element-loop fallback (plan §3, D7, milestone M3).

When ``vectorize(f, fallback=True)`` (or ``strict=False``) cannot vectorize
``f``, the function is wrapped in an element-wise loop over its Array API
arguments. This is a correctness escape hatch, not vectorized code: it is
slow and emits a ``UserWarning`` when used.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

from array_api_compat import array_namespace

from .compat import _is_array, _py_scalar

__all__ = ["make_fallback"]


def make_fallback(func: Callable[..., Any], reason: str) -> Callable[..., Any]:
    """Wrap ``func`` in an element loop over its array arguments."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        positions = [i for i, a in enumerate(args) if _is_array(a)]
        kw_positions = [k for k, v in kwargs.items() if _is_array(v)]
        if not positions and not kw_positions:
            raise TypeError(
                f"vectorized fallback for {func.__name__!r} requires at least one "
                "Array API array argument"
            )
        arrays = [args[i] for i in positions] + [kwargs[k] for k in kw_positions]
        xp = array_namespace(*arrays)
        broadcast = xp.broadcast_arrays(*arrays)
        shape = broadcast[0].shape
        flat = [xp.reshape(a, (-1,)) for a in broadcast]
        out = []
        for elems in zip(*flat, strict=True):
            call_args = list(args)
            call_kwargs = dict(kwargs)
            values = iter(elems)
            for pos in positions:
                call_args[pos] = _py_scalar(next(values))
            for key in kw_positions:
                call_kwargs[key] = _py_scalar(next(values))
            out.append(func(*call_args, **call_kwargs))
        return xp.reshape(xp.asarray(out), shape)

    wrapper.source = (  # type: ignore[attr-defined]
        f"# fallback element-loop wrapper around {func.__name__!r}\n"
        f"# (source-to-source vectorization failed: {reason})"
    )
    wrapper._vectorized_original = func  # type: ignore[attr-defined]
    wrapper.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
    return wrapper
