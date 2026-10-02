"""vectorize(f): compile a scalar Python function into an Array API vectorized function."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

from ._codegen import generate_source
from ._errors import VectorizationError
from ._extract import extract_function
from ._fallback import make_fallback
from ._lower import lower_function
from ._optimize import optimize
from ._runtime import compile_vectorized
from ._validate import validate

__version__ = "0.1.0"

__all__ = ["VectorizationError", "get_source", "vectorize"]

#: memoized vectorized helpers, keyed by the original function object (D7)
_HELPER_CACHE: dict[Callable[..., Any], Callable[..., Any]] = {}
#: functions currently being vectorized (recursion detection)
_ACTIVE_HELPERS: set[Callable[..., Any]] = set()


def _vectorize_strict(func: Callable[..., Any]) -> Callable[..., Any]:
    """The core pipeline: extract -> validate -> lower -> optimize -> codegen -> compile."""
    info = extract_function(func)
    validate(info)
    lowered = lower_function(info, helper_vectorizer=_vectorize_helper)
    program = optimize(lowered.program, user_names=info.user_names)
    source = generate_source(lowered, program)
    return compile_vectorized(
        source, lowered.name, lowered.hidden_params, original=func, helpers=lowered.helpers
    )


def _vectorize_helper(callee: Callable[..., Any]) -> Callable[..., Any]:
    if callee in _HELPER_CACHE:
        return _HELPER_CACHE[callee]
    if callee in _ACTIVE_HELPERS:
        raise VectorizationError(
            f"cannot vectorize {callee.__name__!r}: recursive helper calls are not supported"
        )
    _ACTIVE_HELPERS.add(callee)
    try:
        vec = _vectorize_strict(callee)
    finally:
        _ACTIVE_HELPERS.discard(callee)
    _HELPER_CACHE[callee] = vec
    return vec


def vectorize(
    func: Callable[..., Any],
    *,
    strict: bool = True,
    fallback: bool = False,
) -> Callable[..., Any]:
    """Compile an inspectable scalar function into a vectorized one.

    The returned callable runs over any Array API backend (NumPy, PyTorch,
    JAX, CuPy, array-api-strict, ...) using only standard functions. The
    generated source is available as ``.source`` and via ``inspect.getsource``.

    Raises :class:`VectorizationError` (never silently miscompiles) when the
    function uses constructs outside the supported subset. With
    ``fallback=True`` (or ``strict=False``), a failing function is instead
    wrapped in an element-wise loop with a ``UserWarning``.
    """
    try:
        return _vectorize_strict(func)
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
