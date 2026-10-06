"""Opt-in element-loop fallback.

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

from array_vectorize.codegen.docstring import prefixed_summary
from array_vectorize.compat import _check_namespace, _is_array, _py_scalar

__all__ = ["make_fallback"]


def make_fallback(
    func: Callable[..., Any], reason: str, namespace: Any = None
) -> Callable[..., Any]:
    """Wrap ``func`` in an element loop over its array arguments.

    With ``namespace`` set (``vectorize(..., namespace=...)``) the loop uses
    it directly instead of extracting the namespace from the arguments.
    """

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
        xp = namespace if namespace is not None else array_namespace(*arrays)
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

    def with_namespace(xp: Any) -> Callable[..., Any]:
        _check_namespace(xp)
        # cheap to build: no caching; the new wrapper gets its own
        # with_namespace, so pins chain
        return make_fallback(func, reason, namespace=xp)

    wrapper.with_namespace = with_namespace  # type: ignore[attr-defined]
    # functools.wraps copied the original's __doc__ verbatim; re-prefix its
    # summary so help() marks the wrapper as vectorized like the strict
    # results are (the scalar source stays reachable via the original —
    # it is often unavailable here, which is a common fallback reason).
    # A pathological non-str __doc__ is left exactly as wraps left it.
    if isinstance(func.__doc__, str):
        wrapper.__doc__ = prefixed_summary(func.__doc__)
    wrapper.source = (  # type: ignore[attr-defined]
        f"# fallback element-loop wrapper around {func.__name__!r}\n"
        f"# (source-to-source vectorization failed: {reason})"
    )
    wrapper._vectorized_original = func  # type: ignore[attr-defined]
    wrapper.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
    return wrapper
