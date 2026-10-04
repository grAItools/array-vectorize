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


def _vec_minmax(xp: Any, is_min: Any, *args: Any) -> Any:
    """Exact-semantics minimum/maximum (injected into generated modules).

    Selects per Python scalar semantics rather than backend promotion:
    - identical array dtypes with exactly-fitting (or never-winning,
      clamped) literal bounds keep that dtype;
    - mixed int/float promotes to float64, all-float uses the widest
      float dtype;
    - pure-integer mixes with uint64 use proven result ranges: max
      results always fit uint64 (a signed value only wins when positive,
      so negatives clamp to 0); min results fit int64 whenever something
      signed can win (uint64 values above int64-max can never win a min
      against a signed value, so they clamp to int64-max);
    - other integer mixes widen to the containing signed dtype.

    Strict backends require identical array dtypes for xp.minimum /
    xp.maximum and reject cross-kind xp.result_type, so the selection is
    computed here, in Python, from the runtime dtypes.
    """
    fn = xp.minimum if is_min else xp.maximum

    def fold(cast: list[Any]) -> Any:
        # minimum/maximum are binary in the Array API: fold pairwise
        result = cast[0]
        for c in cast[1:]:
            result = fn(result, c)
        return result

    # boolean arrays normalize to int8: strict backends reject bool
    # operands in minimum/maximum (and in the comparisons below), and
    # 0/1 int8 values are exact for every boolean selection
    args = tuple(
        xp.astype(a, xp.int8) if hasattr(a, "dtype") and "bool" in str(a.dtype) else a for a in args
    )
    arrays = [a for a in args if hasattr(a, "dtype")]
    lits = [a for a in args if not hasattr(a, "dtype")]

    if arrays:
        first_dt = arrays[0].dtype
        if all(a.dtype == first_dt for a in arrays):
            if all(_fits_dtype(b, first_dt) for b in lits):
                cast = [a if hasattr(a, "dtype") else xp.asarray(a, dtype=first_dt) for a in args]
                return fold(cast)
            if all(_minmax_bound_kind(b, is_min, first_dt) != "promote" for b in lits):
                cast = []
                for a in args:
                    if hasattr(a, "dtype"):
                        cast.append(a)
                    elif _fits_dtype(a, first_dt):
                        cast.append(xp.asarray(a, dtype=first_dt))
                    else:  # clamp: the bound can never win
                        bound = 2 ** _dtype_bits(first_dt) - 1 if is_min else 0
                        cast.append(xp.asarray(bound, dtype=first_dt))
                return fold(cast)

    has_float = any("float" in str(a.dtype) or "complex" in str(a.dtype) for a in arrays) or any(
        isinstance(b, float) for b in lits
    )
    if has_float:
        classes = [_describe_dtype(a) for a in args]
        if any(not f for f, _ in classes):
            dt = xp.float64
        else:
            dt = getattr(xp, f"float{max(w for f, w in classes if f)}")
        return fold([xp.asarray(a, dtype=dt) for a in args])

    # pure-integer domain
    u64_present = any("uint" in str(a.dtype) and _dtype_bits(a.dtype) >= 64 for a in arrays)
    negative_capable = any(
        "int" in str(a.dtype) and "uint" not in str(a.dtype) for a in arrays
    ) or any(isinstance(b, int) and not isinstance(b, bool) and b < 0 for b in lits)
    if u64_present:
        if is_min and negative_capable:
            imax = 2**63 - 1
            cast = []
            for a in args:
                if hasattr(a, "dtype"):
                    if "uint" in str(a.dtype) and _dtype_bits(a.dtype) >= 64:
                        cast.append(
                            xp.astype(
                                xp.where(a > imax, xp.asarray(imax, dtype=a.dtype), a),
                                xp.int64,
                            )
                        )
                    else:
                        cast.append(xp.astype(a, xp.int64))
                else:
                    cast.append(xp.asarray(a, dtype=xp.int64))
            return fold(cast)
        cast = []
        for a in args:
            if hasattr(a, "dtype"):
                if "uint" in str(a.dtype):
                    cast.append(xp.astype(a, xp.uint64))
                else:
                    cast.append(
                        xp.astype(xp.where(a < 0, xp.asarray(0, dtype=a.dtype), a), xp.uint64)
                    )
            elif isinstance(a, int) and not isinstance(a, bool) and a < 0:
                cast.append(xp.asarray(0, dtype=xp.uint64))
            else:
                cast.append(xp.asarray(a, dtype=xp.uint64))
        return fold(cast)
    classes = [_describe_dtype(a) for a in args]
    dt = getattr(xp, f"int{max(w for _, w in classes)}")
    return fold([xp.asarray(a, dtype=dt) for a in args])


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
