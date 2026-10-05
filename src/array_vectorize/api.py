"""Public entry points: ``vectorize``, ``get_source``, and the helper cache.

Option handling and the fallback decision live here; the strict compilation
stages live in ``pipeline.compile_function``. This module owns the global
helper cache: memoized vectorized helpers are keyed by
``(function object, protect flag, namespace)`` and are never evicted — a
protected or pinned compilation must not reuse a helper compiled with
different options (and vice versa), and helper recursion is detected via
``_ACTIVE_HELPERS``. It also owns ``_PIN_CACHE``, memoizing the strict
``with_namespace`` variants per ``(original, protect, namespace)``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

from .compat import _check_namespace
from .errors import VectorizationError
from .fallback import make_fallback
from .pipeline import compile_function

__all__ = ["get_source", "vectorize"]

#: memoized vectorized helpers, keyed by (function object, protect, namespace)
#: (D7): pin and protect options must not leak between cached helpers
_HELPER_CACHE: dict[tuple[Callable[..., Any], bool, Any], Callable[..., Any]] = {}
#: functions currently being vectorized (recursion detection)
_ACTIVE_HELPERS: set[Callable[..., Any]] = set()
#: memoized strict ``with_namespace`` variants, keyed by
#: (scalar original, protect, namespace) — identity-stable across calls
_PIN_CACHE: dict[tuple[Callable[..., Any], bool, Any], Callable[..., Any]] = {}


def _vectorize_strict(
    func: Callable[..., Any],
    *,
    protect: bool = False,
    verify_args: tuple[Any, ...] | None = None,
    namespace: Any = None,
) -> Callable[..., Any]:
    """Run the pipeline with helper vectorization wired to the api cache."""
    return compile_function(
        func,
        protect=protect,
        verify_args=verify_args,
        namespace=namespace,
        # helpers inherit the caller's protect_domains setting AND pin
        helper_vectorizer=lambda callee: _vectorize_helper(
            callee, protect=protect, namespace=namespace
        ),
    )


def _vectorize_helper(
    callee: Callable[..., Any], *, protect: bool = False, namespace: Any = None
) -> Callable[..., Any]:
    # keyed by (function, protect_domains, namespace): a protected or pinned
    # compilation must not reuse a helper compiled with different options
    key = (callee, protect, namespace)
    if key in _HELPER_CACHE:
        return _HELPER_CACHE[key]
    if callee in _ACTIVE_HELPERS:
        raise VectorizationError(
            f"cannot vectorize {callee.__name__!r}: recursive helper calls are not supported"
        )
    _ACTIVE_HELPERS.add(callee)
    try:
        vec = _vectorize_strict(callee, protect=protect, namespace=namespace)
    finally:
        _ACTIVE_HELPERS.discard(callee)
    _HELPER_CACHE[key] = vec
    return vec


def _attach_namespace_api(vec: Callable[..., Any], *, protect: bool) -> None:
    """Attach ``with_namespace(xp)`` to a strict vectorized result.

    ``with_namespace`` returns a NEW pinned callable (never mutates ``vec``),
    memoized in ``_PIN_CACHE`` so repeated pins are identity-stable; the
    variant itself gets the API attached, so pins chain. Re-compiling via
    ``_vectorize_strict`` (not ``vectorize``) skips ``verify=`` — body
    codegen is unchanged, only the ``xp`` binding differs.
    """

    def with_namespace(xp: Any) -> Callable[..., Any]:
        _check_namespace(xp)
        # chains to the true scalar original, so pins compose
        original: Callable[..., Any] = vec._vectorized_original  # type: ignore[attr-defined]
        key = (original, protect, xp)
        if key not in _PIN_CACHE:
            variant = _vectorize_strict(original, protect=protect, namespace=xp)
            _attach_namespace_api(variant, protect=protect)
            _PIN_CACHE[key] = variant
        return _PIN_CACHE[key]

    vec.with_namespace = with_namespace  # type: ignore[attr-defined]


def vectorize(
    func: Callable[..., Any],
    *,
    strict: bool = True,
    fallback: bool = False,
    protect_domains: bool = False,
    verify: tuple[Any, ...] | None = None,
    namespace: Any = None,
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

    ``namespace=xp`` pins the array namespace at decoration time: the
    generated code binds ``xp`` directly to it and never extracts the
    namespace from its arguments — all-scalar calls become legal. Passing
    arguments compatible with the pinned namespace is the USER's
    responsibility. ``vec.with_namespace(xp)`` returns a NEW callable
    pinned to ``xp`` (``.source``, ``.__signature__`` and
    ``._vectorized_original`` are preserved; identical pins return the
    same object). An invalid namespace raises ``TypeError``.
    """
    if namespace is not None:
        _check_namespace(namespace)  # usage error: fail fast, before compiling
    try:
        vec = _vectorize_strict(
            func, protect=protect_domains, verify_args=verify, namespace=namespace
        )
    except VectorizationError as exc:
        if fallback or not strict:
            reason = str(exc).splitlines()[0]
            warnings.warn(
                f"vectorize({getattr(func, '__name__', repr(func))!r}): falling back to an "
                f"element-wise loop ({reason}); this is not vectorized code",
                UserWarning,
                stacklevel=2,
            )
            return make_fallback(func, reason, namespace=namespace)
        raise
    # strict results (not internally compiled helpers) carry with_namespace
    _attach_namespace_api(vec, protect=protect_domains)
    return vec


def get_source(vectorized: Callable[..., Any]) -> str:
    """Return the generated source of a vectorized function."""
    source = getattr(vectorized, "source", None)
    if not isinstance(source, str):
        raise VectorizationError(f"{vectorized!r} is not a vectorized function")
    return source
