"""Exact-semantics min/max helper (from _runtime).

``_vec_minmax`` is injected into generated modules and executes on the
user's backend at call time.
"""

from __future__ import annotations

from typing import Any

from .dtype import _describe_dtype, _dtype_bits, _fits_dtype, _minmax_bound_kind

__all__ = ["_vec_minmax"]


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
