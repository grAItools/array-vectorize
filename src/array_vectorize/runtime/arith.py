"""Exact-semantics arithmetic helper (from _runtime).

``_vec_arith`` and its uint64-exactness primitives are injected into
generated modules and execute on the user's backend at call time.
"""

from __future__ import annotations

import enum
import operator
from typing import Any

from array_vectorize.runtime import dtype

__all__ = [
    "_ARITH_FNS",
    "_ARITH_OPS",
    "ArithOp",
    "_is_u64",
    "_to_u64",
    "_u64_sub_exact",
    "_vec_arith",
    "u64_dividend",
]


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
    return hasattr(a, "dtype") and "uint" in str(a.dtype) and dtype._dtype_bits(a.dtype) >= 64


def _to_u64(xp: Any, a: Any) -> Any:
    """Cast to uint64, wrapping negative int literals modularly.

    NumPy rejects out-of-bounds Python ints in asarray.
    """
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
    """Exact-semantics arithmetic for bool-intified operations.

    Injected into generated modules; ``op_id`` indexes _ARITH_OPS.

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
    classes = [dtype._describe_dtype(a) for a in args]
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
