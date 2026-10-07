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


def _integer_true_divide(xp: Any, left: Any, right: Any) -> Any:
    """Correctly round ratios of 64-bit integers without rounding inputs.

    Binary long division collects 53 significand bits plus guard/sticky
    bits. Signed int64 stores unsigned magnitudes modularly, allowing
    backends without uint64 arithmetic to use the same path. Remainders
    are doubled via subtraction so comparisons never depend on overflow.
    The fixed trip count also permits tracing by JAX and PyTorch.
    """
    left_array = xp.asarray(left)
    right_array = xp.asarray(right)
    a = xp.astype(left_array, xp.int64)
    b = xp.astype(right_array, xp.int64)
    l_neg = a < 0 if "uint" not in str(left_array.dtype) else xp.asarray(False)
    r_neg = b < 0 if "uint" not in str(right_array.dtype) else xp.asarray(False)
    a = xp.where(l_neg, -a, a)
    b = xp.where(r_neg, -b, b)
    zero_divisor = b == 0
    b = xp.where(zero_divisor, xp.asarray(1, dtype=xp.int64), b)
    remainder = xp.zeros_like(a + b)
    mantissa = xp.zeros_like(remainder)
    count = xp.zeros_like(remainder)
    exponent = xp.zeros_like(remainder)
    guard = xp.zeros_like(remainder, dtype=xp.bool)
    sticky = xp.zeros_like(guard)
    for position in range(63, -119, -1):
        incoming = (a >> position) & 1 if position >= 0 else xp.asarray(0, dtype=xp.int64)
        threshold = b - remainder - incoming
        # Unsigned comparison using signed storage: a set sign bit means
        # the unsigned value is larger than every nonnegative int64.
        bit = xp.where((remainder < 0) != (threshold < 0), remainder < 0, remainder >= threshold)
        remainder = xp.where(
            bit, remainder - (b - remainder) + incoming, remainder + remainder + incoming
        )
        started = (count != 0) | bit
        exponent = xp.where((count == 0) & bit, xp.asarray(position), exponent)
        collect = started & (count < 53)
        mantissa = xp.where(collect, mantissa * 2 + xp.astype(bit, xp.int64), mantissa)
        rounding_bit = count == 53
        guard = xp.where(rounding_bit, bit, guard)
        unconsumed = (a & ((1 << position) - 1)) != 0 if position > 0 else xp.asarray(False)
        sticky = xp.where(rounding_bit, (remainder != 0) | unconsumed, sticky)
        count = xp.where(started, count + 1, count)
    round_up = guard & (sticky | ((mantissa & 1) != 0))
    mantissa = mantissa + xp.astype(round_up, xp.int64)
    result = xp.astype(mantissa, xp.float64) * xp.pow(
        xp.asarray(2.0, dtype=xp.float64), xp.astype(exponent - 52, xp.float64)
    )
    result = xp.where(l_neg != r_neg, -result, result)
    # Numeric exceptions deliberately become IEEE values (design D3).
    denominator = xp.where(
        zero_divisor, xp.asarray(0.0, dtype=xp.float64), xp.asarray(1.0, dtype=xp.float64)
    )
    return result / denominator


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
    - array % negative-literal and array // negative-literal: magnitudes
      are computed exactly in uint64, then negated into int64 when they
      fit. Quotients below int64-min use the float64 fallback;
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
    if op == "div" and not has_real_float:
        return _integer_true_divide(xp, left, right)
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
                if bool(xp.all(mag <= 2**63)):
                    return -xp.astype(mag, xp.int64)
                # Larger negative quotients cannot fit int64. Preserve
                # their sign with the lattice's float64 fallback.
                return xp.where(
                    mag == 0, xp.asarray(0.0, dtype=xp.float64), -xp.astype(mag, xp.float64)
                )
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
