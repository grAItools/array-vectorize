"""Compile/exec/linecache plumbing and wrapper attributes (plan §9)."""

from __future__ import annotations

import inspect
import linecache
import struct
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


def _fits_dtype(value: Any, dtype: Any) -> bool:
    """True when a raw Python scalar is EXACTLY representable in ``dtype``.

    Floats are checked by an IEEE round trip at the dtype's width (float32
    cannot hold 16777217); ints by range; bools only in bool dtypes.
    """
    name = str(dtype)
    if "bool" in name:
        return isinstance(value, bool)
    digits = "".join(d for d in name if d.isdigit())
    if "float" in name:
        code = {16: "e", 32: "f", 64: "d"}.get(int(digits) if digits else 64)
        if code is None:
            return False  # exotic float width: promote instead of risking rounding
        try:
            packed = struct.unpack("<" + code, struct.pack("<" + code, value))[0]
            return bool(packed == value)
        except (OverflowError, ValueError, struct.error):
            return False
    try:
        v = int(value)
    except (TypeError, ValueError, OverflowError):
        return False
    if v != value:  # e.g. a float literal for an int dtype
        return False
    bits = int(digits) if digits else 64
    if "uint" in name:
        return bool(0 <= v < 2**bits)
    return bool(-(2 ** (bits - 1)) <= v < 2 ** (bits - 1))


def _dtype_bits(dtype: Any) -> int:
    digits = "".join(d for d in str(dtype) if d.isdigit())
    return int(digits) if digits else 64


def _minmax_bound_kind(lit: Any, is_min: Any, dtype: Any) -> str:
    """ "fit", "clamp", or "promote" for a literal min/max bound.

    A bound that cannot win is CLAMPED into an unsigned array dtype
    instead of forcing promotion: max(unsigned, negative) never selects
    the bound, and min(unsigned, >= 2**bits) never selects it either —
    replacing it with the dtype's own extreme preserves the exact array
    values (float64 cannot hold every uint64).
    """
    if _fits_dtype(lit, dtype):
        return "fit"
    name = str(dtype)
    # max with a negative bound, or min with a too-large bound
    if (
        "uint" in name
        and isinstance(lit, int)
        and not isinstance(lit, bool)
        and ((not is_min and lit < 0) or (is_min and lit >= 2 ** _dtype_bits(dtype)))
    ):
        return "clamp"
    return "promote"


def _vec_minmax_dtype(xp: Any, is_min: Any, *args: Any) -> Any:
    """Common dtype for min/max arguments (injected into generated modules).

    min/max results are always one of the input values, so no arithmetic
    headroom is needed. Fast path: when every ARRAY argument shares one
    actual dtype and every raw literal either fits it exactly or clamps
    (can never win), that dtype is used unchanged — keeping uint64 pairs
    and fitting bounds exact. Slow path: class-based promotion — mixed
    int/float promotes to float64 (Python semantics), ints widen to the
    largest width present, uint64 counts as float64 (no signed int holds
    it). Strict backends additionally require identical array dtypes for
    xp.minimum/xp.maximum and reject cross-kind xp.result_type, so the
    common dtype is computed here.
    """
    arrays = [a for a in args if hasattr(a, "dtype")]
    lits = [a for a in args if not hasattr(a, "dtype")]
    if arrays:
        first_dt = arrays[0].dtype
        if all(a.dtype == first_dt for a in arrays) and all(
            _minmax_bound_kind(b, is_min, first_dt) != "promote" for b in lits
        ):
            return first_dt
    classes = [_describe_dtype(a) for a in args]
    has_float = any(f for f, _ in classes)
    has_int = any(not f for f, _ in classes)
    if has_float:
        if has_int:
            return xp.float64
        return getattr(xp, f"float{max(w for _, w in classes)}")
    return getattr(xp, f"int{max(w for _, w in classes)}")


def _vec_minmax_lit(xp: Any, is_min: Any, lit: Any, *rest: Any) -> Any:
    """A literal min/max bound as an array (injected into generated modules).

    ``rest`` carries the OTHER arguments (literals and arrays), so the
    final common dtype accounts for every literal: a fitting bound still
    casts to float64 when another bound forces promotion. Against the
    final dtype the bound is cast exactly when representable, clamped
    when it can never win (max with a negative bound on unsigned values
    -> 0; min with a too-large bound -> uint-max), else promoted.
    """
    arrays = [a for a in rest if hasattr(a, "dtype")]
    other_lits = [a for a in rest if not hasattr(a, "dtype")]
    common = _vec_minmax_dtype(xp, is_min, lit, *other_lits, *arrays)
    if _fits_dtype(lit, common):
        return xp.asarray(lit, dtype=common)
    if (
        arrays
        and len({a.dtype for a in arrays}) == 1
        and common == arrays[0].dtype
        and _minmax_bound_kind(lit, is_min, common) == "clamp"
    ):
        bound = 2 ** _dtype_bits(common) - 1 if is_min else 0
        return xp.asarray(bound, dtype=common)
    return xp.asarray(lit, dtype=common)


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
