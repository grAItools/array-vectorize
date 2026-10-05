"""Public entry points: ``vectorize``, ``get_source``, and the helper cache.

Option handling and the fallback decision live here; the strict compilation
stages live in ``pipeline.compile_function``. This module owns the global
helper cache: memoized vectorized helpers are keyed by
``(function object, protect flag)`` and are never evicted — a protected
compilation must not reuse an unprotected helper compiled earlier (and vice
versa), and helper recursion is detected via ``_ACTIVE_HELPERS``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

from .errors import VectorizationError
from .fallback import make_fallback
from .pipeline import compile_function

__all__ = ["get_source", "vectorize"]

#: memoized vectorized helpers, keyed by the original function object (D7)
_HELPER_CACHE: dict[tuple[Callable[..., Any], bool], Callable[..., Any]] = {}
#: functions currently being vectorized (recursion detection)
_ACTIVE_HELPERS: set[Callable[..., Any]] = set()


def _vectorize_strict(
    func: Callable[..., Any],
    *,
    protect: bool = False,
    verify_args: tuple[Any, ...] | None = None,
) -> Callable[..., Any]:
    """Run the pipeline with helper vectorization wired to the api cache."""
    return compile_function(
        func,
        protect=protect,
        verify_args=verify_args,
        # helpers inherit the caller's protect_domains setting
        helper_vectorizer=lambda callee: _vectorize_helper(callee, protect=protect),
    )


def _vectorize_helper(callee: Callable[..., Any], *, protect: bool = False) -> Callable[..., Any]:
    # keyed by (function, protect_domains): a protected compilation must
    # not reuse an unprotected helper compiled earlier (and vice versa)
    key = (callee, protect)
    if key in _HELPER_CACHE:
        return _HELPER_CACHE[key]
    if callee in _ACTIVE_HELPERS:
        raise VectorizationError(
            f"cannot vectorize {callee.__name__!r}: recursive helper calls are not supported"
        )
    _ACTIVE_HELPERS.add(callee)
    try:
        vec = _vectorize_strict(callee, protect=protect)
    finally:
        _ACTIVE_HELPERS.discard(callee)
    _HELPER_CACHE[key] = vec
    return vec


def vectorize(
    func: Callable[..., Any],
    *,
    strict: bool = True,
    fallback: bool = False,
    protect_domains: bool = False,
    verify: tuple[Any, ...] | None = None,
) -> Callable[..., Any]:
    """Compile an inspectable scalar function into a vectorized one.

    The returned callable runs over any Array API backend (NumPy, PyTorch,
    JAX, CuPy, array-api-strict, ...) using only standard functions. The
    generated source is available as ``.source`` and via ``inspect.getsource``.

    Raises :class:`VectorizationError` (never silently miscompiles) when the
    function uses constructs outside the supported subset. With
    ``fallback=True`` (or ``strict=False``), a failing function is instead
    wrapped in an element-wise loop with a ``UserWarning``.

    ``protect_domains=True`` clamps partial-function arguments (sqrt, log,
    asin, ...) inside where-branches so dead lanes never leave the domain:
    results never change, warnings disappear, a few extra ops are added.

    ``verify=example_args`` differentially checks the generated function
    against the scalar original on the given inputs at generation time.
    """
    try:
        return _vectorize_strict(func, protect=protect_domains, verify_args=verify)
    except VectorizationError as exc:
        if fallback or not strict:
            reason = str(exc).splitlines()[0]
            warnings.warn(
                f"vectorize({getattr(func, '__name__', repr(func))!r}): falling back to an "
                f"element-wise loop ({reason}); this is not vectorized code",
                UserWarning,
                stacklevel=2,
            )
            return make_fallback(func, reason)
        raise


def get_source(vectorized: Callable[..., Any]) -> str:
    """Return the generated source of a vectorized function."""
    source = getattr(vectorized, "source", None)
    if not isinstance(source, str):
        raise VectorizationError(f"{vectorized!r} is not a vectorized function")
    return source
