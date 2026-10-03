"""Compile/exec/linecache plumbing and wrapper attributes (plan §9)."""

from __future__ import annotations

import inspect
import linecache
from collections.abc import Callable
from typing import Any, cast

from ._ir import generated_name

__all__ = ["compile_vectorized"]

_COMMON_DTYPE_NAME = "_vec_common_dtype"


def _vec_common_dtype(xp: Any, *args: Any) -> Any:
    """Common dtype for min/max arguments (injected into generated modules).

    Python semantics promote mixed int/float to float and widen ints to
    fit; strict backends additionally require identical array dtypes for
    ``xp.minimum``/``xp.maximum`` (their ``result_type`` rejects
    cross-kind promotion). Every argument is cast to the returned dtype.
    """
    best_rank = 0  # 0 = int-ish (bool/int/uint), 1 = float
    best_width = 1
    best_is_float = False
    for a in args:
        if hasattr(a, "dtype"):
            name = str(a.dtype)
        elif isinstance(a, float):
            name = "float64"
        else:
            v = int(a)
            if -(2**7) <= v < 2**7:
                name = "int8"
            elif -(2**15) <= v < 2**15:
                name = "int16"
            elif -(2**31) <= v < 2**31:
                name = "int32"
            else:
                name = "int64"
        is_float = "float" in name or "complex" in name
        digits = [d for d in name if d.isdigit()]
        width = int("".join(digits)) if digits else (64 if is_float else 8)
        if "uint" in name:
            # unsigned ints do not fit the same-width signed dtype; numpy
            # promotes uint8 + int8 to int16, so approximate with width * 2
            width = min(width * 2, 64)
        rank = 1 if is_float else 0
        if (rank, width) > (best_rank, best_width):
            best_rank, best_width, best_is_float = rank, width, is_float
    return getattr(xp, f"{'float' if best_is_float else 'int'}{best_width}")


def compile_vectorized(
    source: str,
    name: str,
    hidden_params: list[tuple[str, Any]],
    original: Callable[..., Any],
    helpers: list[tuple[str, Any]] | None = None,
) -> Callable[..., Any]:
    """Compile the generated module and return the callable itself.

    - compiles as ``<vectorizer:{name}>`` and registers the source in
      ``linecache`` with ``mtime=None`` (survives ``checkcache``), so
      ``inspect.getsource`` and tracebacks show the real generated lines;
    - injects closure-array defaults into ``__kwdefaults__`` and vectorized
      helper functions into the module namespace;
    - sets ``.source`` and the ``_vectorized_original`` marker.
    """
    filename = f"<vectorizer:{name}>"
    code = compile(source, filename, "exec")
    namespace: dict[str, Any] = {}
    exec(code, namespace)
    # retrieve the entry point BEFORE injecting helpers, so a helper whose
    # generated name collides can never shadow the function itself
    func: Any = namespace[generated_name(name)]
    # runtime dtype helper for min/max argument promotion (bare-name calls
    # resolve through the function's globals at call time)
    namespace[_COMMON_DTYPE_NAME] = _vec_common_dtype
    for helper_name, helper_fn in helpers or []:
        namespace[helper_name] = helper_fn

    if hidden_params:
        # merge: replacing __kwdefaults__ would erase the function's own
        # keyword-only defaults
        merged = {**getattr(func, "__kwdefaults__", {})}
        merged.update(dict(hidden_params))
        func.__kwdefaults__ = merged

    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)

    func.__name__ = original.__name__
    func.__qualname__ = getattr(original, "__qualname__", original.__name__)
    # NOTE: __wrapped__ is intentionally NOT set: inspect.getsourcelines
    # unwraps unconditionally, which would hide the generated source. The
    # signature is preserved via __signature__ and the original is reachable
    # via the _vectorized_original marker (see _extract).
    func.__signature__ = inspect.signature(original)
    func._vectorized_original = original
    func.source = source
    return cast(Callable[..., Any], func)
