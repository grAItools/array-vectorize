"""Direct unit tests for the call-time runtime library (RESTRUCTURE.md §6).

runtime/arith, runtime/minmax and runtime/dtype are normally exercised
only end-to-end through generated modules; these tests pin their
exact-semantics contracts per lane, cross-checked against plain Python
scalar semantics in the comments.
"""

from __future__ import annotations

import numpy as np
import numpy as xp

from array_vectorize.runtime.arith import ArithOp, _vec_arith
from array_vectorize.runtime.dtype import _describe_dtype, _fits_dtype
from array_vectorize.runtime.minmax import _vec_minmax

U64_MAX = 2**64 - 1
I64_MAX = 2**63 - 1


# ------------------------------------------------------------------ arith


def test_vec_arith_bool_add_is_exact_int64() -> None:
    # Python: True + True == 2, True + False == 1, False + True == 1
    a = np.asarray([True, True, False])
    b = np.asarray([True, False, True])
    out = _vec_arith(xp, int(ArithOp.ADD), a, b)
    assert out.dtype == np.dtype("int64")
    assert np.array_equal(out, [2, 1, 1])


def test_vec_arith_uint64_add_bool_is_modular_uint64() -> None:
    # Python ints are unbounded; uint64 is the widest container, so add is
    # modular past 2**64 (documented backend behavior): (2**64-1) + 1 == 0
    a = np.asarray([3, U64_MAX], dtype=np.uint64)
    b = np.asarray([True, True])
    out = _vec_arith(xp, int(ArithOp.ADD), a, b)
    assert out.dtype == np.dtype("uint64")
    assert np.array_equal(out, [4, 0])


def test_vec_arith_uint64_sub_crossing_zero_is_exact_int64() -> None:
    # Python: 5-7 == -2, 3-1 == 2, (2**64-1)-(2**63) == 2**63-1
    left = np.asarray([5, 3, U64_MAX], dtype=np.uint64)
    right = np.asarray([7, 1, 2**63], dtype=np.uint64)
    out = _vec_arith(xp, int(ArithOp.SUB), left, right)
    assert out.dtype == np.dtype("int64")
    assert np.array_equal(out, [-2, 2, I64_MAX])


def test_vec_arith_uint64_sub_all_nonnegative_stays_uint64() -> None:
    # Python: 7-5 == 2, 3-1 == 2, 10-2 == 8 — all non-negative
    left = np.asarray([7, 3, 10], dtype=np.uint64)
    right = np.asarray([5, 1, 2], dtype=np.uint64)
    out = _vec_arith(xp, int(ArithOp.SUB), left, right)
    assert out.dtype == np.dtype("uint64")
    assert np.array_equal(out, [2, 2, 8])


def test_vec_arith_uint64_mod_negative_literal_is_exact_int64() -> None:
    # Python: 10 % -3 == -2, 7 % -3 == -2, (2**64-1) % -3 == 0, 0 % -3 == 0
    # (sign follows the divisor; 2**64-1 is divisible by 3)
    a = np.asarray([10, 7, U64_MAX, 0], dtype=np.uint64)
    out = _vec_arith(xp, int(ArithOp.MOD), a, -3)
    assert out.dtype == np.dtype("int64")
    assert np.array_equal(out, [-2, -2, 0, 0])


def test_vec_arith_uint64_floordiv_negative_literal_is_exact_int64() -> None:
    # Python: 10 // -3 == -4, 7 // -3 == -3, (2**64-1) // -3 == -6148914691236517205,
    # 0 // -3 == 0 (floor division rounds toward minus infinity)
    a = np.asarray([10, 7, U64_MAX, 0], dtype=np.uint64)
    out = _vec_arith(xp, int(ArithOp.FLOORDIV), a, -3)
    assert out.dtype == np.dtype("int64")
    assert np.array_equal(out, [-4, -3, -6148914691236517205, 0])


def test_vec_arith_div_is_always_float() -> None:
    # true division is float even for all-integer operands (strict
    # backends require floating-point operands)
    out_i = _vec_arith(xp, int(ArithOp.DIV), np.asarray([1, 7, 9]), np.asarray([2, 2, 2]))
    assert out_i.dtype == np.dtype("float64")
    assert np.array_equal(out_i, [0.5, 3.5, 4.5])
    out_u = _vec_arith(
        xp,
        int(ArithOp.DIV),
        np.asarray([9, 3], dtype=np.uint64),
        np.asarray([3, 2], dtype=np.uint64),
    )
    assert out_u.dtype == np.dtype("float64")
    assert np.array_equal(out_u, [3.0, 1.5])


def test_vec_arith_neg_uint64_fits_int64() -> None:
    # Python: -5, -0 == 0 — magnitudes up to 2**63 fit int64 exactly
    out = _vec_arith(xp, int(ArithOp.NEG), np.asarray([5, 0], dtype=np.uint64))
    assert out.dtype == np.dtype("int64")
    assert np.array_equal(out, [-5, 0])


# ----------------------------------------------------------------- minmax


def test_vec_minmax_uint64_signed_min_fits_int64() -> None:
    # Python: min(2**64-1, -1) == -1, min(5, 3) == 3, min(100, -100) == -100;
    # uint64 lanes above int64-max can never win a min against a signed
    # value, so they clamp to int64-max
    u = np.asarray([U64_MAX, 5, 100], dtype=np.uint64)
    s = np.asarray([-1, 3, -100], dtype=np.int64)
    out = _vec_minmax(xp, True, u, s)
    assert out.dtype == np.dtype("int64")
    assert np.array_equal(out, [-1, 3, -100])


def test_vec_minmax_uint64_signed_max_clamps_negatives() -> None:
    # Python: max(2**64-1, -1) == 2**64-1, max(5, 3) == 5, max(100, -100) == 100;
    # negative signed lanes can never win a max against an unsigned value,
    # so they clamp to 0 and the result stays uint64
    u = np.asarray([U64_MAX, 5, 100], dtype=np.uint64)
    s = np.asarray([-1, 3, -100], dtype=np.int64)
    out = _vec_minmax(xp, False, u, s)
    assert out.dtype == np.dtype("uint64")
    assert np.array_equal(out, [U64_MAX, 5, 100])


def test_vec_minmax_bool_only_selection_returns_bool() -> None:
    # Python: min(True, False) == False, min(False, False) == False,
    # min(True, True) == True; max(True, True) == True
    a = np.asarray([True, False, True])
    b = np.asarray([False, False, True])
    out = _vec_minmax(xp, True, a, b)
    assert out.dtype == np.dtype("bool")
    assert np.array_equal(out, [False, False, True])
    out_lit = _vec_minmax(xp, False, np.asarray([True, False]), True)
    assert out_lit.dtype == np.dtype("bool")
    assert np.array_equal(out_lit, [True, True])


# ------------------------------------------------------------------- dtype


def test_fits_dtype_edge_cases() -> None:
    # float32 cannot represent 2**24+1 (rounds to 16777216.0)
    assert not _fits_dtype(16777217, np.dtype("float32"))
    assert _fits_dtype(16777216, np.dtype("float32"))
    # -1 is outside [0, 2**64)
    assert not _fits_dtype(-1, np.dtype("uint64"))
    assert _fits_dtype(-1, np.dtype("int64"))
    # only actual bools fit a bool dtype
    assert _fits_dtype(True, np.dtype("bool"))
    assert not _fits_dtype(2, np.dtype("bool"))


def test_describe_dtype_classifies_uint64_as_float64_class() -> None:
    # uint64 fits no signed int, so it promotes as float64: (True, 64)
    assert _describe_dtype(np.asarray([1], dtype=np.uint64)) == (True, 64)
    assert _describe_dtype(1.5) == (True, 64)
    assert _describe_dtype(np.asarray([1], dtype=np.int32)) == (False, 32)
    # small int literals count as the minimal dtype that fits
    assert _describe_dtype(5) == (False, 8)
    assert _describe_dtype(np.asarray([True])) == (False, 8)
