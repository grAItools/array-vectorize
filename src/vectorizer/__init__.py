"""vectorize(f): compile a scalar Python function into an Array API vectorized function."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

from ._errors import VectorizationError
from ._fallback import make_fallback
from ._lower import lower_function
from ._runtime import compile_vectorized
from ._verify import verify_match
from .codegen import generate_source
from .frontend.extract import extract_function
from .frontend.validate import validate

# NOTE: aliased twice-over — binding `optimize` here would clobber the
# `vectorizer.optimize` package attribute with the pipeline function, and
# aliasing to `_optimize` would collide with the `_optimize.py` compat shim
# (importing the shim rebinds that attribute with the module)
from .optimize import optimize as optimize_
from .optimize import protect_domains

__version__ = "0.1.0"

__all__ = ["VectorizationError", "get_source", "vectorize"]

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
    """The core pipeline: extract -> validate -> lower -> optimize -> codegen -> compile."""
    info = extract_function(func)
    validate(info)
    lowered = lower_function(
        info,
        # helpers inherit the caller's protect_domains setting
        helper_vectorizer=lambda callee: _vectorize_helper(callee, protect=protect),
    )
    program = optimize_(lowered.program, user_names=info.user_names | lowered.emitted_names)
    if protect:
        program = protect_domains(program)
    source = generate_source(lowered, program)
    vec = compile_vectorized(
        source, lowered.name, lowered.hidden_params, original=func, helpers=lowered.helpers
    )
    if verify_args is not None:
        verify_match(vec, func, verify_args)
    return vec


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
