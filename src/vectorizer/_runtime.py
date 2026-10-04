"""Compile/exec/linecache plumbing and wrapper attributes (plan §9)."""

from __future__ import annotations

import enum
import inspect
import linecache
import operator
import struct
from collections.abc import Callable
from typing import Any, cast

from ._ir import generated_name

__all__ = ["ArithOp", "compile_vectorized"]


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

    orig_arrays = [a for a in args if hasattr(a, "dtype")]
    lits = [a for a in args if not hasattr(a, "dtype")]
    had_bool = bool(orig_arrays) and any("bool" in str(a.dtype) for a in orig_arrays)
    bool_only = (
        had_bool
        and all("bool" in str(a.dtype) for a in orig_arrays)
        and all(isinstance(b, bool) for b in lits)
    )
    # boolean arrays normalize to int8: strict backends reject bool
    # operands in minimum/maximum (and in the comparisons below), and
    # 0/1 int8 values are exact for every boolean selection
    args = tuple(
        xp.astype(a, xp.int8) if hasattr(a, "dtype") and "bool" in str(a.dtype) else a for a in args
    )
    arrays = [a for a in args if hasattr(a, "dtype")]

    if bool_only:
        # boolean-only selection: compute in int8, restore bool so
        # downstream boolean operations keep boolean semantics
        cast = [a if hasattr(a, "dtype") else xp.asarray(int(a), dtype=xp.int8) for a in args]
        return xp.astype(fold(cast), xp.bool)
    if (
        had_bool
        and arrays
        and all(a.dtype == xp.int8 for a in arrays)
        and not any(isinstance(b, float) for b in lits)
    ):
        # every array was boolean (normalized to int8) with non-boolean
        # operands: select in int64 — the compiler introduced the bool->
        # int conversion, and Python's integer semantics are unbounded,
        # so downstream arithmetic must not overflow int8
        return fold([xp.asarray(a, dtype=xp.int64) for a in args])

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


class ArithOp(enum.IntEnum):
    """Stable ids for the ops accepted by _vec_arith.

    Generated code passes the id as an int literal. Lowering must convert
    to a plain int at the Literal boundary (``Literal(int(ArithOp.ADD),
    "int")``): ``ast.unparse`` on an enum value would emit its repr and
    corrupt the generated source.
    """

    NEG = 0
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    FLOORDIV = 5
    MOD = 6


#: operator names accepted by _vec_arith, in ArithOp id order
#: (generated code passes the id as an int literal)
_ARITH_OPS = tuple(m.name.lower() for m in ArithOp)

_ARITH_FNS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
    "floordiv": operator.floordiv,
    "mod": operator.mod,
}


def _is_u64(a: Any) -> bool:
    return hasattr(a, "dtype") and "uint" in str(a.dtype) and _dtype_bits(a.dtype) >= 64


def _to_u64(xp: Any, a: Any) -> Any:
    """Cast to uint64, wrapping negative int literals modularly (NumPy
    rejects out-of-bounds Python ints in asarray)."""
    if isinstance(a, int) and not isinstance(a, bool) and a < 0:
        a = a % 2**64
    return xp.asarray(a, dtype=xp.uint64)


def _u64_sub_exact(xp: Any, left: Any, right: Any) -> Any | None:
    """Result-aware subtraction of integer operands in uint64 range.

    Python ints keep their true sign here (arrays on these paths are
    non-negative: uint64/bool, or signed arrays gated non-negative).
    Returns uint64 when the difference is provably non-negative, or
    the exact int64 difference when every per-lane result fits int64
    (int64 subtraction is mod-2**64, so wrapped uint64 casts
    reinterpret exactly). None signals results that fit no single
    dtype: the caller falls back to the documented float64.
    """
    l_neg = isinstance(left, int) and not isinstance(left, bool) and left < 0
    r_neg = isinstance(right, int) and not isinstance(right, bool) and right < 0
    lu = _to_u64(xp, left)
    ru = _to_u64(xp, right)
    if r_neg:
        # left - (negative) = left + |right|: always non-negative
        # (modular uint64; overflow past 2**64 is the documented
        # backend behavior, same as add)
        return lu - ru
    if l_neg:
        # (negative) - right = -(|left| + right): always negative;
        # exact in int64 when the magnitude fits
        if -left <= 2**63 and bool(xp.all(ru <= 2**63 + left)):
            return xp.astype(lu, xp.int64) - xp.astype(ru, xp.int64)
        return None
    ge = lu >= ru
    if bool(xp.all(ge)):
        return lu - ru  # every result in [0, 2**64)
    pos = xp.where(ge, lu - ru, xp.asarray(0, dtype=lu.dtype))
    neg = xp.where(ge, xp.asarray(0, dtype=lu.dtype), ru - lu)
    if bool(xp.all(pos <= 2**63 - 1)) and bool(xp.all(neg <= 2**63)):
        return xp.astype(lu, xp.int64) - xp.astype(ru, xp.int64)
    return None


def _vec_arith(xp: Any, op_id: Any, left: Any, right: Any = None) -> Any:
    """Exact-semantics arithmetic for bool-intified operations (injected
    into generated modules). ``op_id`` indexes _ARITH_OPS.

    Bools are cast to a numeric dtype (int64 headroom; Python integers
    are unbounded). Floats keep their dtype; mixed int/float promotes to
    float64. True division always computes in floats (strict backends
    require floating-point operands).

    uint64 integer paths are EXACT (no real float involved, other
    operands unsigned/boolean arrays or int literals):
    - add/sub/mul and floordiv/mod with non-negative divisors: modular
      uint64 (exact while the true result is in [0, 2**64) — all any
      dtype can hold);
    - array % negative-literal and array // negative-literal: the results
      ALWAYS fit int64 (remainder magnitude < |divisor| <= 2**63; floor
      magnitude <= 2**63), so the magnitudes are computed exactly in
      uint64 and negated into int64 — per lane, regardless of how large
      the input values are;
    - negative-literal % array and negation: int64 whenever the values
      fit (checked at runtime); otherwise float64, because those results
      are genuinely unrepresentable in int64 on the offending lanes.
    """
    op = _ARITH_OPS[op_id] if isinstance(op_id, int) else op_id
    fn: Any = _ARITH_FNS.get(op, operator.neg)
    args = [left] + ([right] if right is not None else [])
    has_real_float = any(
        hasattr(a, "dtype") and ("float" in str(a.dtype) or "complex" in str(a.dtype)) for a in args
    ) or any(isinstance(a, float) for a in args)
    if any(_is_u64(a) for a in args) and not has_real_float:
        others_intish = all(
            (hasattr(a, "dtype") and ("uint" in str(a.dtype) or "bool" in str(a.dtype)))
            or isinstance(a, bool | int)
            for a in args
        )
        u64s = [a for a in args if _is_u64(a)]
        if not others_intish and op in ("add", "sub", "mul", "floordiv", "mod"):
            # signed ARRAY operands: exact when a single integer dtype can
            # hold the operands — int64 when every uint64 value fits
            # (checked at runtime; handles negative results exactly), or
            # modular uint64 when every signed value is
            # non-negative. Mixed-sign per-lane results (a huge uint64
            # plus a negative signed value) fit no single dtype: the
            # lattice's float64 is the documented best effort.
            signed_arrs = [
                a
                for a in args
                if hasattr(a, "dtype")
                and "int" in str(a.dtype)
                and "uint" not in str(a.dtype)
                and "bool" not in str(a.dtype)
            ]
            if signed_arrs:
                if all(bool(xp.all(u <= 2**63 - 1)) for u in u64s):
                    return fn(*[xp.asarray(a, dtype=xp.int64) for a in args])
                if all(bool(xp.all(s >= 0)) for s in signed_arrs):
                    # non-negative signed values are faithful in uint64:
                    # add/mul are modular-exact, and %,// match Python
                    # for non-negative dividends and divisors. Subtraction
                    # is result-aware (see _u64_sub_exact): mixed-magnitude
                    # differences fit no single dtype and fall through to
                    # the documented float64.
                    if op == "sub":
                        res = _u64_sub_exact(xp, left, right)
                        if res is not None:
                            return res
                    else:
                        return fn(*[_to_u64(xp, a) for a in args])
        if others_intish:
            if op in ("add", "mul"):
                return fn(_to_u64(xp, left), _to_u64(xp, right))
            if op == "sub":
                # result-aware: representable negative differences must
                # not wrap (None falls through to the float64 lattice)
                res = _u64_sub_exact(xp, left, right)
                if res is not None:
                    return res
            if op in ("floordiv", "mod") and right is not None and u64_dividend(left, right):
                # array op negative-literal: exact in int64 per lane
                d = xp.asarray(-right, dtype=left.dtype)
                if op == "mod":
                    # a % -|d| is 0 when a %% |d| == 0, else (a %% |d|) + d;
                    # adding the NEGATIVE divisor directly avoids forming
                    # |d| = 2**63, which no int64 scalar can hold
                    r = xp.astype(left % d, xp.int64)
                    return xp.where(r == 0, xp.asarray(0, dtype=xp.int64), r + right)
                q = left // d
                r = left % d
                mag = q + xp.astype(r > 0, left.dtype)  # ceil(a / |d|)
                return -xp.astype(mag, xp.int64)
            if op == "mod" and isinstance(left, int) and left < 0 and _is_u64(right):
                # negative-literal % array: (v - |d| mod v) mod v — always
                # in [0, v), exactly representable in uint64
                d = xp.asarray(-left, dtype=right.dtype)
                return (right - (d % right)) % right
            if op == "floordiv" and isinstance(left, int) and left < 0 and _is_u64(right):
                # negative-literal // array: -ceil(|d| / v), exact in int64
                d = xp.asarray(-left, dtype=right.dtype)
                mag = (d // right) + xp.astype((d % right) > 0, right.dtype)
                return -xp.astype(mag, xp.int64)
            if (
                op in ("floordiv", "mod")
                and right is not None
                and all(
                    isinstance(a, bool) or (isinstance(a, int) and a >= 0)
                    for a in args
                    if not hasattr(a, "dtype")
                )
            ):
                return fn(_to_u64(xp, left), _to_u64(xp, right))
            if op == "neg":
                if bool(xp.all(left <= 2**63)):
                    return -xp.astype(left, xp.int64)
                return -xp.astype(left, xp.float64)
        if op == "div":
            return operator.truediv(
                xp.asarray(left, dtype=xp.float64), xp.asarray(right, dtype=xp.float64)
            )
    # generic: cast to the common dtype and compute
    classes = [_describe_dtype(a) for a in args]
    has_float = any(f for f, _ in classes)
    has_int = any(not f for f, _ in classes)
    if op == "div" or (has_float and has_int):
        dt: Any = xp.float64
    elif has_float:
        dt = getattr(xp, f"float{max(w for f, w in classes)}")
    else:
        dt = xp.int64
    if right is None:
        return -xp.asarray(left, dtype=dt)
    return fn(xp.asarray(left, dtype=dt), xp.asarray(right, dtype=dt))


def u64_dividend(left: Any, right: Any) -> bool:
    """True for the array-op-negative-literal forms (array on the left)."""
    return isinstance(right, int) and not isinstance(right, bool) and right < 0 and _is_u64(left)


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
