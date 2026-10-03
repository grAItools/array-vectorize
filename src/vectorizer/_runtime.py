"""Compile/exec/linecache plumbing and wrapper attributes (plan §9)."""

from __future__ import annotations

import inspect
import linecache
from collections.abc import Callable
from typing import Any, cast

from ._ir import generated_name

__all__ = ["compile_vectorized"]


def _describe_dtype(value: Any) -> tuple[bool, int]:
    """Classify a value for promotion: (is_float, width).

    Arrays contribute their dtype; float literals count as float64; int
    literals count as the minimal dtype that fits. Unsigned ints do not
    fit the same-width signed dtype (uint8 + int8 promotes to int16), and
    uint64 fits no signed int at all, so it counts as float64.
    """
    if hasattr(value, "dtype"):
        name = str(value.dtype)
    elif isinstance(value, float):
        name = "float64"
    else:
        v = int(value)
        if -(2**7) <= v < 2**7:
            name = "int8"
        elif -(2**15) <= v < 2**15:
            name = "int16"
        elif -(2**31) <= v < 2**31:
            name = "int32"
        else:
            name = "int64"
    is_float = "float" in name or "complex" in name
    digits = "".join(d for d in name if d.isdigit())
    width = int(digits) if digits else (64 if is_float else 8)
    if "uint" in name and not is_float:
        if width >= 64:
            return True, 64  # uint64: only float64 can hold it
        width = min(width * 2, 64)
    return is_float, width


def _vec_common_dtype(xp: Any, *args: Any) -> Any:
    """Common dtype for min/max arguments (injected into generated modules).

    min/max results are always one of the input values, so no arithmetic
    headroom is needed: identical argument dtypes pass through unchanged,
    mixed int/float promotes to float64 (Python semantics), and ints widen
    to the largest width present. Strict backends additionally require
    identical array dtypes for xp.minimum/xp.maximum and reject cross-kind
    xp.result_type, so the common dtype is computed here.
    """
    classes = [_describe_dtype(a) for a in args]
    first = next((a for a in args if hasattr(a, "dtype")), None)
    if first is not None and all(c == classes[0] for c in classes):
        return first.dtype
    has_float = any(f for f, _ in classes)
    has_int = any(not f for f, _ in classes)
    if has_float:
        if has_int:
            return xp.float64
        return getattr(xp, f"float{max(w for _, w in classes)}")
    return getattr(xp, f"int{max(w for _, w in classes)}")


def _vec_arith_dtype(xp: Any, *args: Any) -> Any:
    """Common dtype for bool-intified arithmetic (injected into generated
    modules).

    Python integers are unbounded, so int-class operands get int64
    headroom (True + True chains must not wrap); float operands keep their
    dtype (bools are exact in any of them); mixed int/float promotes to
    float64, matching Python's int + float -> float.
    """
    classes = [_describe_dtype(a) for a in args]
    has_float = any(f for f, _ in classes)
    has_int = any(not f for f, _ in classes)
    if has_float:
        if has_int:
            return xp.float64
        return getattr(xp, f"float{max(w for _, w in classes)}")
    return xp.int64


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
    # helper functions (vectorized user helpers AND the runtime dtype
    # promotion helpers) are injected here; bare-name calls resolve through
    # the function's globals at call time
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
