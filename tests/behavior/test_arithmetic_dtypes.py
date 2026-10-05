"""Bool/int/uint64 arithmetic exactness: dtype promotion, literals, min/max, bitwise, negation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from support import make_fn, make_module, vec_of

from vectorizer import vectorize

spec = importlib.util.spec_from_file_location(
    "vec_corpus_b", Path(__file__).parent.parent / "corpus.py"
)
assert spec is not None and spec.loader is not None
CORPUS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CORPUS)


def vfn(name: str) -> Any:
    return vectorize(getattr(CORPUS, name))


X = np.asarray([-2.0, -0.5, 0.0, 0.5, 2.0, 10.0])
Y = np.asarray([3.0, 1.5, 1.0, 0.5, -1.0, -3.0])


def py_scalar(got: Any) -> int:
    return int(np.asarray(got).reshape(-1)[0])


# ---- from test_behavior (git history: tests/test_behavior.py)


def test_add() -> None:
    assert np.allclose(vfn("add")(X, Y), X + Y)


def test_arith_ops() -> None:
    expected = (X + Y - X * Y) / (X * X + Y * Y)
    assert np.allclose(vfn("arith_ops")(X, Y), expected)


def test_bitwise_ops() -> None:
    xi = np.asarray([-3, 2, 5, 7], dtype=np.int64)
    yi = np.asarray([4, 1, 6, 2], dtype=np.int64)
    expected = (xi & yi) | (xi ^ 3) << 1 >> 2
    assert np.array_equal(vfn("bitwise_ops")(xi, yi), expected)


def test_unary_ops_int() -> None:
    xi = np.asarray([-3, 2, 5], dtype=np.int64)
    assert np.array_equal(vfn("unary_ops")(xi), -xi + (+xi) - (~xi))


def test_minmax() -> None:
    expected = np.minimum(X, Y) + np.maximum(np.maximum(X, Y), 3.0)
    assert np.allclose(vfn("minmax")(X, Y), expected)


def test_compare_chain() -> None:
    got = vfn("compare_chain")(X)
    assert got.dtype == np.bool_
    assert np.array_equal(got, (X > 0) & (X < 1))


def test_boolop_logical() -> None:
    got = vfn("boolop_logical")(X, Y)
    expected = ((X > 0) & (Y > 0)) | ~(X < 0)
    assert np.array_equal(got, expected)


def test_boolop_numeric_exact_including_nan() -> None:
    arr = np.asarray([-0.0, 0.0, 2.0, np.nan])
    got = vfn("boolop_numeric")(arr, np.asarray([5.0, 5.0, 0.0, 1.0]))
    # Python: x and y -> x if falsy else y; NaN is truthy
    expected = np.asarray([arr[0], 0.0, 0.0, 1.0])
    assert np.array_equal(got, expected, equal_nan=True)


def test_clamp() -> None:
    assert np.allclose(vfn("clamp")(X), np.clip(X, 0.0, 1.0))
    assert np.allclose(vfn("clamp")(X, -1.0, 0.5), np.clip(X, -1.0, 0.5))
    assert np.allclose(vfn("clamp")(X, hi=0.25), np.clip(X, 0.0, 0.25))


def test_scalar_python_args_accepted() -> None:
    # Python scalars ride along via standard promotion
    assert np.allclose(vfn("add")(X, 2.0), X + 2.0)


# ---- from test_review_round1 (git history: tests/test_review_round1.py)


@pytest.mark.parametrize(
    "body",
    [
        "    b = x > 0\n    return b + b",
        "    b = x > 0\n    b += x > 0\n    return b",
        "    b = x > 0\n    c = x > 0\n    return b * c + b",
    ],
)
def test_boolean_arithmetic_through_assignments(body: str) -> None:
    vec, fn = vec_of(body)
    xs = np.asarray([-1.0, 1.0])
    assert list(map(int, vec(xs))) == [int(fn(float(v))) for v in xs]


# ---- from test_review_round3 (git history: tests/test_review_round3.py)


def test_abs_exact_large_int() -> None:
    vec = vectorize(make_fn("    return x + abs(9007199254740993)"))
    got = vec(np.asarray([0], dtype=np.int64))
    assert got[0] == 9007199254740993


def test_abs_int_bitwise() -> None:
    vec = vectorize(make_fn("    return abs(3) & x"))
    got = vec(np.asarray([1], dtype=np.int64))
    assert got[0] == 1


def test_unary_negate_bool() -> None:
    # -True is -1 in Python; backends cannot negate bool arrays
    vec = vectorize(make_fn("    return -(x > 0)"))
    assert vec(np.asarray([2.0]))[0] == -1


def test_pow_negative_exponent_int_base() -> None:
    # Python promotes int ** negative-int to float; arrays raise instead
    vec = vectorize(make_fn("    return x ** -1"))
    assert np.allclose(vec(np.asarray([2.0, 4.0])), [0.5, 0.25])


def test_pow_negative_exponent_literal_base() -> None:
    vec = vectorize(make_fn("    return 2 ** -1"))
    got = vec(np.asarray([0.0]))
    # constant functions fold to a scalar result; broadcast for comparison
    assert np.allclose(np.broadcast_to(np.asarray(got), (1,)), [0.5])


def test_minmax_all_literals_fold() -> None:
    vec = vectorize(make_fn("    return min(3, 5) + x"))
    assert np.allclose(vec(np.asarray([1.0])), [4.0])


def test_minmax_loop_var_sibling() -> None:
    # the loop variable is a raw Python int per iteration: min/max must
    # wrap it (asarray) and give int literals a matching dtype
    vec = vectorize(
        make_fn("    m = x\n    for i in range(4):\n        m = min(i, 2) + m\n    return m")
    )
    # min(i, 2) over i = 0..3 sums to 0 + 1 + 2 + 2 = 5
    assert vec(np.asarray([0.0]))[0] == 5.0


# ---- from test_review_round4 (git history: tests/test_review_round4.py)


def test_bool_plus_large_int_exact() -> None:
    vec = vectorize(make_fn("    return (x > 0) + 9007199254740992"))
    got = vec(np.asarray([1.0]))
    assert got[0] == 9007199254740993


def test_bool_arith_then_bitwise() -> None:
    vec = vectorize(make_fn("    return ((x > 0) + (x > 0)) & 1"))
    got = vec(np.asarray([1.0]))
    assert got[0] == 0


def test_min_literal_exact_large_int() -> None:
    vec = vectorize(make_fn("    return min(x, 9007199254740993)"))
    got = vec(np.asarray([9007199254740994], dtype=np.int64))
    assert got[0] == 9007199254740993


# ---- from test_review_round5 (git history: tests/test_review_round5.py)


def test_min_float_literal_int_input() -> None:
    # the float bound must widen the common dtype, not truncate to int64
    vec = vectorize(make_fn("    return min(x, 1.5)"))
    got = vec(np.asarray([2], dtype=np.int64))
    assert got[0] == 1.5


def test_max_literal_int8_no_overflow() -> None:
    # 300 must widen int8, not wrap to 44
    vec = vectorize(make_fn("    return max(x, 300)"))
    got = vec(np.asarray([1], dtype=np.int8))
    assert got[0] == 300


def test_min_compound_sibling_exact_big_int() -> None:
    vec = vectorize(make_fn("    return min(x + 0, 9007199254740993)"))
    got = vec(np.asarray([9007199254740994], dtype=np.int64))
    assert got[0] == 9007199254740993


def test_bool_bitwise_stays_bool() -> None:
    vec = vectorize(make_fn("    b = (x > 0) & (x < 3)\n    return b + b"))
    got = vec(np.asarray([1.0]))
    assert got[0] == 2


def test_bool_bitwise_or_xor_stay_bool() -> None:
    vec = vectorize(make_fn("    return ((x > 0) | (x > 1)) + ((x > 0) ^ (x > 1))"))
    got = vec(np.asarray([1.0]))
    assert got[0] == 2


# ---- from test_review_round6 (git history: tests/test_review_round6.py)


def test_bool_arith_times_hundred() -> None:
    vec = vectorize(make_fn("    return ((x > 0) + (x > 0)) * 100"))
    assert vec(np.asarray([1.0]))[0] == 200


def test_min_uint64_vs_literal() -> None:
    vec = vectorize(make_fn("    return min(x, 1)"))
    got = vec(np.asarray([2**63], dtype=np.uint64))
    assert got[0] == 1


def test_min_float32_vs_int64_exact() -> None:
    vec = vectorize(make_fn("    return min(x, y)"))
    got = vec(np.asarray([20000000], dtype=np.float32), np.asarray([16777217], dtype=np.int64))
    assert got[0] == 16777217


def test_minmax_identical_dtypes_passthrough() -> None:
    vec = vectorize(make_fn("    return min(x, y)"))
    got = vec(np.asarray([3, 1], dtype=np.int8), np.asarray([2, 2], dtype=np.int8))
    assert list(np.asarray(got)) == [2, 1]


def test_bool_plus_int8_array_headroom() -> None:
    vec = vectorize(make_fn("    return (x > 0) + y", defaults="x, y"))
    got = vec(np.asarray([1.0]), np.asarray([100], dtype=np.int8))
    assert got[0] == 101


# ---- from test_review_round7 (git history: tests/test_review_round7.py)


def test_max_uint8_int16_no_narrowing() -> None:
    # uint8 and int16 share the promotion class (int, 16): the fast path
    # must not pick uint8 and wrap 300 to 44
    vec = vectorize(make_fn("    return max(x, y)"))
    got = vec(np.asarray([1], dtype=np.uint8), np.asarray([300], dtype=np.int16))
    assert got[0] == 300


def test_min_uint8_int16_negative() -> None:
    vec = vectorize(make_fn("    return min(x, y)"))
    got = vec(np.asarray([1], dtype=np.uint8), np.asarray([-1], dtype=np.int16))
    assert got[0] == -1


def test_max_bool_array_vs_literal() -> None:
    # bool and the literal 2 share the promotion class (int, 8): the fast
    # path must not pick bool and collapse 2 to True
    vec = vectorize(make_fn("    return max(x, 2)"))
    got = vec(np.asarray([False]))
    assert got[0] == 2


def test_min_uint64_vs_float64() -> None:
    # uint64 and float64 share the promotion class (float, 64): the fast
    # path must not pick uint64 and truncate 1.5 to 1
    vec = vectorize(make_fn("    return min(x, y)"))
    got = vec(np.asarray([2**63], dtype=np.uint64), np.asarray([1.5]))
    assert got[0] == 1.5


def test_max_uint64_literal_exact() -> None:
    # the literal 1 fits uint64, so the fast path keeps uint64: the
    # result is exact, not the float64 rounding of 9223372036854775809
    vec = vectorize(make_fn("    return max(x, 1)"))
    got = vec(np.asarray([9223372036854775809], dtype=np.uint64))
    assert got[0] == 9223372036854775809


def test_min_float32_literal_needs_wider_dtype() -> None:
    # 16777217 is not exactly representable in float32: the fast path
    # must reject it and promote to float64
    vec = vectorize(make_fn("    return min(x, 16777217)"))
    got = vec(np.asarray([3e7], dtype=np.float32))
    assert got[0] == 16777217


def test_min_float32_exact_literal_keeps_float32() -> None:
    vec = vectorize(make_fn("    return min(x, 1.5)"))
    got = vec(np.asarray([2.0], dtype=np.float32))
    assert got[0] == 1.5


def test_min_int_literal_fit_boundaries() -> None:
    vec = vectorize(make_fn("    return min(x, 127)"))
    assert vec(np.asarray([100], dtype=np.int8))[0] == 100
    vec = vectorize(make_fn("    return max(x, 128)"))
    # 128 does not fit int8: promotion widens instead of wrapping
    assert vec(np.asarray([1], dtype=np.int8))[0] == 128


# ---- from test_review_round8 (git history: tests/test_review_round8.py)


def test_max_float16_literal_precision() -> None:
    # 2049 is not representable in float16: the fast path must reject it
    # and promote to float64 instead of rounding the bound to 2048
    vec = vectorize(make_fn("    return max(x, 2049.0)"))
    got = vec(np.asarray([0.0], dtype=np.float16))
    assert got[0] == 2049.0


def test_max_float16_exact_literal_keeps_dtype() -> None:
    vec = vectorize(make_fn("    return max(x, 2048.0)"))
    got = vec(np.asarray([0.0], dtype=np.float16))
    assert got[0] == 2048.0


def test_max_uint64_negative_bound_exact() -> None:
    # a negative bound never wins a max over unsigned values: it is
    # clamped into the unsigned dtype instead of forcing float64 (which
    # cannot hold 9223372036854775809 exactly)
    vec = vectorize(make_fn("    return max(x, -1)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64))
    # exact comparison via Python int (numpy scalar == rounds floats)
    assert int(got[0]) == 9223372036854775809


def test_min_uint64_negative_bound_literal_wins() -> None:
    # min selects the negative bound: it must survive exactly
    vec = vectorize(make_fn("    return min(x, -1)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64))
    assert int(got[0]) == -1


def test_min_uint8_too_large_bound_clamps() -> None:
    # 300 never wins a min over uint8 values (max 255): clamp to 255
    vec = vectorize(make_fn("    return min(x, 300)"))
    got = vec(np.asarray([1], dtype=np.uint8))
    assert got[0] == 1


def test_max_uint8_negative_bound_clamps() -> None:
    vec = vectorize(make_fn("    return max(x, -1)"))
    got = vec(np.asarray([7], dtype=np.uint8))
    assert got[0] == 7


# ---- from test_review_round9 (git history: tests/test_review_round9.py)


def test_max_uint64_negative_literal_exact() -> None:
    # `-1` lowers to UnaryOp(neg, 1): it must be recognized as a literal
    # so the never-winning-bound clamp fires (float64 would round
    # 9223372036854775809 to ...808)
    vec = vectorize(make_fn("    return max(x, -1)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64))
    # exact comparison via Python int (numpy scalar == rounds floats)
    assert int(got[0]) == 9223372036854775809


def test_min_uint64_negative_literal_wins_exactly() -> None:
    vec = vectorize(make_fn("    return min(x, -1)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64))
    assert int(got[0]) == -1


def test_negative_literal_other_contexts() -> None:
    vec = vectorize(make_fn("    return min(x, -1) + max(y, -2.5)"))
    got = vec(np.asarray([3.0]), np.asarray([-3.0]))
    assert got[0] == -3.5  # min(3, -1) + max(-3, -2.5) = -1 + -2.5


def test_min_variadic_negative_and_float_literals() -> None:
    vec = vectorize(make_fn("    return min(x, -1, 1.5)"))
    got = vec(np.asarray([2], dtype=np.uint8))
    assert int(got[0]) == -1


def test_max_variadic_fitting_literals_keep_dtype() -> None:
    vec = vectorize(make_fn("    return max(x, 1, 2)"))
    got = vec(np.asarray([0], dtype=np.uint8))
    assert int(got[0]) == 2


# ---- from test_review_round10 (git history: tests/test_review_round10.py)


def test_min_uint64_always_winning_literal_exact() -> None:
    # the negative bound always wins: the result is the literal, exactly
    # representable in int64 (float64 would round -9007199254740993)
    vec = vectorize(make_fn("    return min(x, -9007199254740993)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64))
    assert py_scalar(got) == -9007199254740993


def test_min_uint64_result_supports_bitwise() -> None:
    # the min result must stay INTEGER so downstream int ops work
    vec = vectorize(make_fn("    return min(x, -1) & 1"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64))
    assert py_scalar(got) == 1


def test_max_uint64_int64_arrays_exact() -> None:
    vec = vectorize(make_fn("    return max(x, y)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64), np.asarray([-1], dtype=np.int64))
    assert py_scalar(got) == 2**63 + 1


def test_min_uint64_int64_arrays_exact() -> None:
    vec = vectorize(make_fn("    return min(x, y)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64), np.asarray([-1], dtype=np.int64))
    assert py_scalar(got) == -1


def test_min_uint64_signed_positive_literals() -> None:
    # min results fit int64 when something signed can win
    vec = vectorize(make_fn("    return min(x, -1, 5)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64))
    assert py_scalar(got) == -1


def test_max_uint64_all_unsigned_widens() -> None:
    vec = vectorize(make_fn("    return max(x, y)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64), np.asarray([7], dtype=np.uint8))
    assert py_scalar(got) == 2**63 + 1


def test_min_max_roundtrip_prior_findings() -> None:
    # the whole prior promotion matrix stays exact after the restructure
    vec = vectorize(make_fn("    return max(x, -1)"))
    assert py_scalar(vec(np.asarray([2**63 + 1], dtype=np.uint64))) == 9223372036854775809
    vec = vectorize(make_fn("    return max(x, 1)"))
    assert py_scalar(vec(np.asarray([9223372036854775809], dtype=np.uint64))) == 9223372036854775809
    vec = vectorize(make_fn("    return min(x, 300)"))
    assert py_scalar(vec(np.asarray([1], dtype=np.uint8))) == 1
    vec = vectorize(make_fn("    return max(x, y)"))
    assert py_scalar(vec(np.asarray([1], dtype=np.uint8), np.asarray([300], dtype=np.int16))) == 300
    vec = vectorize(make_fn("    return min(x, 1.5)"))
    assert vec(np.asarray([2], dtype=np.int64))[0] == 1.5
    vec = vectorize(make_fn("    return max(x, 2049.0)"))
    assert vec(np.asarray([0.0], dtype=np.float16))[0] == 2049.0


def test_fit_literal_alongside_clamping_literal() -> None:
    # 5 fits uint8 while -1 clamps: both stay in the shared dtype
    vec = vectorize(make_fn("    return max(x, 5, -1)"))
    got = vec(np.asarray([1], dtype=np.uint8))
    assert py_scalar(got) == 5


def test_max_uint64_mixed_arrays_negative_literal() -> None:
    vec = vectorize(make_fn("    return max(x, y, -1)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64), np.asarray([-5], dtype=np.int64))
    assert py_scalar(got) == 2**63 + 1


def test_max_uint64_mixed_arrays_positive_literal() -> None:
    vec = vectorize(make_fn("    return max(x, y, 3)"))
    got = vec(np.asarray([1], dtype=np.uint64), np.asarray([-5], dtype=np.int64))
    assert py_scalar(got) == 3


def test_min_float16_overflow_literal_promotes() -> None:
    # 1e10 overflows float16 in the IEEE round trip: promote to float64
    vec = vectorize(make_fn("    return min(x, 1e10)"))
    got = vec(np.asarray([0.0], dtype=np.float16))
    assert got[0] == 0.0


def test_fits_dtype_defensive_branches() -> None:
    from vectorizer._runtime import _fits_dtype

    class _FakeDtype:
        def __init__(self, name: str) -> None:
            self._name = name

        def __str__(self) -> str:
            return self._name

    assert not _fits_dtype(1.5, _FakeDtype("float24"))  # exotic width: promote
    assert not _fits_dtype("not-a-number", np.dtype("int64"))
    assert _fits_dtype(True, np.dtype("bool"))
    assert not _fits_dtype(1, np.dtype("bool"))


# ---- from test_review_round11 (git history: tests/test_review_round11.py)


def test_bool_minmax_arithmetic_exact() -> None:
    # min of two bool expressions keeps kind 'bool', so a + a intifies
    vec = vectorize(make_fn("    a = min(x > 0, x > 1)\n    return a + a"))
    assert vec(np.asarray([2]))[0] == 2


def test_bool_minmax_mixed_arithmetic() -> None:
    vec = vectorize(make_fn("    return max(x > 0, x > 1) + (x > 0)"))
    assert vec(np.asarray([2]))[0] == 2


def test_bool_minmax_bitwise_downstream() -> None:
    vec = vectorize(make_fn("    a = max(x > 0, x > 1)\n    return (a + a) & 1"))
    assert vec(np.asarray([2]))[0] == 0


def test_uint64_bool_mix_minmax() -> None:
    # the uint64 paths' signed-clamp comparisons must never see bools
    vec = vectorize(make_fn("    return min(x, y)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64), np.asarray([True]))
    assert int(got[0]) == 1
    vec = vectorize(make_fn("    return max(x, y)"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64), np.asarray([True]))
    assert int(got[0]) == 2**63 + 1


def test_bool_minmax_with_literals() -> None:
    vec = vectorize(make_fn("    return max(x > 0, False)"))
    assert list(np.asarray(vec(np.asarray([2, -1])), dtype=bool)) == [True, False]


# ---- from test_review_round12 (git history: tests/test_review_round12.py)


def test_bool_int_minmax_no_int8_overflow() -> None:
    # min(bool, 1) selects in int64: a * 100 * 2 must not wrap int8
    vec = vectorize(make_fn("    a = min(x > 0, 1)\n    return a * 100 * 2"))
    assert vec(np.asarray([2]))[0] == 200


def test_bool_int_minmax_large_literal() -> None:
    vec = vectorize(make_fn("    a = max(x > 0, 300)\n    return a * 100"))
    assert vec(np.asarray([2]))[0] == 30000


def test_bool_only_minmax_bitwise_numpy() -> None:
    vec = vectorize(make_fn("    return min(x > 0, x > 1) & (x > 0)"))
    # scalar: f(2) = min(True, True) & True = True; f(0) = False & False
    assert list(np.asarray(vec(np.asarray([2, 0])), dtype=bool)) == [True, False]


def test_bool_only_minmax_or_bitwise() -> None:
    vec = vectorize(make_fn("    return max(x > 0, x > 1) | (x > 5)"))
    assert list(np.asarray(vec(np.asarray([2, 0])), dtype=bool)) == [True, False]


def test_bool_only_minmax_still_arithmetics_exact() -> None:
    # bool result + bool result still intifies through kind inference
    vec = vectorize(make_fn("    a = min(x > 0, x > 1)\n    return a + a"))
    assert vec(np.asarray([2]))[0] == 2


def test_bool_float_minmax() -> None:
    vec = vectorize(make_fn("    return min(x > 0, 1.5)"))
    assert vec(np.asarray([2]))[0] == 1.0


# ---- from test_review_round13 (git history: tests/test_review_round13.py)


def test_minmax_runtime_bool_arithmetic_numpy() -> None:
    # x is a boolean array at runtime, but a parameter statically: the
    # min/max result's bool dtype is only known at runtime
    vec = vectorize(make_fn("    a = min(x, True)\n    return a + a"))
    assert list(np.asarray(vec(np.asarray([True, False])))) == [2, 0]


def test_minmax_runtime_bool_ref_chain() -> None:
    vec = vectorize(make_fn("    a = min(x, True)\n    b = a\n    return b + b"))
    assert list(np.asarray(vec(np.asarray([True, False])))) == [2, 0]


def test_minmax_runtime_numeric_noop() -> None:
    # the runtime intify is a no-op for numeric dtypes: min picks x (< 1)
    vec = vectorize(make_fn("    a = min(x, True)\n    return a + a"))
    assert np.allclose(vec(np.asarray([0.5, 0.0])), [1.0, 0.0])
    got = vec(np.asarray([0, -2], dtype=np.int32))
    assert list(np.asarray(got)) == [0, -4]


def test_minmax_runtime_bool_loop_carried() -> None:
    vec = vectorize(
        make_fn("    a = x\n    for i in range(1):\n        a = min(x, True)\n    return a + a")
    )
    assert list(np.asarray(vec(np.asarray([True, False])))) == [2, 0]


def test_minmax_runtime_bool_direct_call() -> None:
    vec = vectorize(make_fn("    return min(x, True) + min(x, True)"))
    assert list(np.asarray(vec(np.asarray([True, False])))) == [2, 0]


# ---- from test_review_round14 (git history: tests/test_review_round14.py)


def test_maybe_bool_uint64_arithmetic_exact() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return a + 0"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64))
    assert int(got[0]) == 9223372036854775809


def test_maybe_bool_uint64_unary_negate_no_rounding_cast() -> None:
    # the arith dtype must not route uint64 through float64
    vec = vectorize(make_fn("    a = max(x, True)\n    return (a + 0) * 1"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64))
    assert int(got[0]) == 9223372036854775809


def test_unary_negate_maybe_bool() -> None:
    vec = vectorize(make_fn("    a = min(x, True)\n    return -a"))
    assert list(np.asarray(vec(np.asarray([True, False])))) == [-1, 0]


# ---- from test_review_round15 (git history: tests/test_review_round15.py)


def test_uint64_negative_literal_arithmetic_exact() -> None:
    # (2**63 + 3) + (-2) = 2**63 + 1: fits uint64; modular uint64
    # arithmetic is exact instead of rounding through float64
    vec = vectorize(make_fn("    a = max(x, True)\n    return a + -2"))
    got = vec(np.asarray([2**63 + 3], dtype=np.uint64))
    assert int(got[0]) == 9223372036854775809


def test_bitwise_maybe_bool_arithmetic() -> None:
    vec = vectorize(make_fn("    a = min(x, True)\n    b = a & True\n    return b + b"))
    assert list(np.asarray(vec(np.asarray([True, False])))) == [2, 0]


def test_bitwise_maybe_bool_or() -> None:
    vec = vectorize(make_fn("    a = min(x, True)\n    b = a | False\n    return b + b"))
    assert list(np.asarray(vec(np.asarray([True, False])))) == [2, 0]


def test_arithmetic_maybe_bool_result_not_maybe_bool() -> None:
    # a + 1 intifies inside, so the result is numeric and needs no
    # second conversion (values stay exact)
    vec = vectorize(make_fn("    a = min(x, True)\n    b = a + 1\n    return b + b"))
    assert list(np.asarray(vec(np.asarray([True, False])))) == [4, 2]


# ---- from test_review_round16 (git history: tests/test_review_round16.py)


def test_uint64_mod_negative_divisor() -> None:
    # -2 % 3 == 1: a negative divisor changes mod semantics, so the
    # uint64 modular path must not wrap the literal
    vec = vectorize(make_fn("    a = max(x, True)\n    return -2 % a"))
    got = vec(np.asarray([3], dtype=np.uint64))
    assert got[0] == 1


def test_uint64_floordiv_negative_divisor() -> None:
    # 3 // -2 == -2: floor division with a negative divisor
    vec = vectorize(make_fn("    a = max(x, True)\n    return a // -2"))
    got = vec(np.asarray([3], dtype=np.uint64))
    assert got[0] == -2.0


def test_uint64_truediv_negative_divisor() -> None:
    # 3 / -2 == -1.5: true division is float, and the operands must be
    # cast to float (strict rejects integer true division of u64 wraps)
    vec = vectorize(make_fn("    a = max(x, True)\n    return a / -2"))
    got = vec(np.asarray([3], dtype=np.uint64))
    assert got[0] == -1.5


def test_uint64_floordiv_mod_positive_keep_uint64() -> None:
    # non-negative divisors keep the exact modular uint64 path
    vec = vectorize(make_fn("    a = max(x, True)\n    return a // 2"))
    got = vec(np.asarray([2**63 + 3], dtype=np.uint64))
    assert int(got[0]) == (2**63 + 3) // 2
    vec = vectorize(make_fn("    a = max(x, True)\n    return a % 2"))
    got = vec(np.asarray([2**63 + 3], dtype=np.uint64))
    assert int(got[0]) == 1


def test_uint64_add_negative_literal_still_modular() -> None:
    # add/sub/mul keep the modular path for negative literals
    vec = vectorize(make_fn("    a = max(x, True)\n    return a + -2"))
    got = vec(np.asarray([2**63 + 3], dtype=np.uint64))
    assert int(got[0]) == 9223372036854775809


# ---- from test_review_round17 (git history: tests/test_review_round17.py)


def test_uint64_mod_negative_large_exact() -> None:
    # (2**53 + 1) % -2 == -1: fits int64; float64 would round the value
    vec = vectorize(make_fn("    a = max(x, True)\n    return a % -2"))
    got = vec(np.asarray([2**53 + 1], dtype=np.uint64))
    assert int(got[0]) == -1


def test_uint64_floordiv_negative_large_exact() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return a // -2"))
    got = vec(np.asarray([2**53 + 1], dtype=np.uint64))
    assert int(got[0]) == -4503599627370497


def test_uint64_negate_large_exact() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return -a"))
    got = vec(np.asarray([2**53 + 1], dtype=np.uint64))
    assert int(got[0]) == -9007199254740993


def test_uint64_negate_huge_best_effort_float() -> None:
    # values beyond int64: the result is unrepresentable exactly; the
    # cast falls back to float64 (documented best effort)
    vec = vectorize(make_fn("    a = max(x, True)\n    return -a"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64))
    assert np.isclose(float(got[0]), -(2.0**63 + 1))


def test_truediv_bool_numpy() -> None:
    vec = vectorize(make_fn("    return min(x, True) / 2"))
    got = vec(np.asarray([True, False]))
    assert list(np.asarray(got)) == [0.5, 0.0]


# ---- from test_review_round18 (git history: tests/test_review_round18.py)


def test_uint64_mod_negative_mixed_lanes_exact() -> None:
    # both remainders fit int64: computed exactly per lane from uint64
    # magnitudes, regardless of input magnitude
    vec = vectorize(make_fn("    return max(x, True) % -2"))
    got = vec(np.asarray([2**53 + 1, 2**63 + 1], dtype=np.uint64))
    assert list(map(int, np.asarray(got))) == [-1, -1]


def test_uint64_floordiv_negative_mixed_lanes_exact() -> None:
    vec = vectorize(make_fn("    return max(x, True) // -2"))
    got = vec(np.asarray([2**53 + 1, 2**63 + 1], dtype=np.uint64))
    assert list(map(int, np.asarray(got))) == [-4503599627370497, -4611686018427387905]


def test_uint64_negate_mixed_lanes_best_effort() -> None:
    # -(2**63 + 1) is unrepresentable in int64: the batch falls back to
    # float64 (documented); the small lane approximates
    vec = vectorize(make_fn("    return -max(x, True)"))
    got = vec(np.asarray([2**53 + 1, 2**63 + 1], dtype=np.uint64))
    assert np.isclose(float(np.asarray(got).reshape(-1)[0]), -9007199254740993)


def test_uint64_negative_literal_mod_array_mixed() -> None:
    # literal-left remainder: (v - |d| mod v) mod v, exact in uint64
    vec = vectorize(make_fn("    return -2 % max(x, True)"))
    got = vec(np.asarray([2**53 + 1, 2**63 + 1], dtype=np.uint64))
    assert list(map(int, np.asarray(got))) == [2**53 - 1, 2**63 - 1]


def test_uint64_negative_literal_floordiv_array() -> None:
    # (-d) // v == -ceil(d / v), exact in int64
    vec = vectorize(make_fn("    return -5 // max(x, True)"))
    got = vec(np.asarray([2], dtype=np.uint64))
    assert int(got[0]) == -3  # -5 // 2 == -3


def test_uint64_add_negative_literal_modular_still_exact() -> None:
    vec = vectorize(make_fn("    return max(x, True) + -2"))
    got = vec(np.asarray([2**63 + 3], dtype=np.uint64))
    assert int(got[0]) == 9223372036854775809


def test_unary_negate_maybe_bool_after_restructure() -> None:
    vec = vectorize(make_fn("    a = min(x, True)\n    return -a"))
    assert list(np.asarray(vec(np.asarray([True, False])))) == [-1, 0]


# ---- from test_review_round19 (git history: tests/test_review_round19.py)


def test_uint64_mod_negative_divisor_formula() -> None:
    # 5 % -3 == -1 (not -(5 % 3) == -2); 6 % -3 == 0
    vec = vectorize(make_fn("    return max(x, True) % -3"))
    assert int(vec(np.asarray([5], dtype=np.uint64))[0]) == -1
    assert int(vec(np.asarray([6], dtype=np.uint64))[0]) == 0


def test_negative_literal_mod_uint64_exact() -> None:
    # -1 % (2**63 + 2) == 2**63 + 1: fits uint64 (not int64)
    vec = vectorize(make_fn("    return -1 % max(x, True)"))
    got = vec(np.asarray([2**63 + 2], dtype=np.uint64))
    assert int(got[0]) == 9223372036854775809


def test_negative_literal_mod_small_array() -> None:
    vec = vectorize(make_fn("    return -2 % max(x, True)"))
    got = vec(np.asarray([2**53 + 1, 2**63 + 1], dtype=np.uint64))
    assert list(map(int, np.asarray(got))) == [2**53 - 1, 2**63 - 1]


# ---- from test_review_round20 (git history: tests/test_review_round20.py)


def test_uint64_mod_int64_min_divisor() -> None:
    mod = make_module(
        "import math\n\n"
        "BOUND = -(2**63)\n\n"
        "def subject(x, y=2.0):\n"
        "    return max(x, True) % BOUND\n"
    )
    vec = vectorize(mod.subject)
    got = vec(np.asarray([1], dtype=np.uint64))
    assert int(got[0]) == -9223372036854775807


def test_arith_result_negative_power() -> None:
    # (x > 0) + 1 is provably int: the negative-exponent float cast
    # must fire on the helper result
    vec = vectorize(make_fn("    a = (x > 0) + 1\n    return a ** -1"))
    assert vec(np.asarray([2]))[0] == 0.5


def test_arith_result_float_kind() -> None:
    # float operands keep float: no int-power issue to begin with
    vec = vectorize(make_fn("    a = (x > 0) + 0.5\n    return a ** -1"))
    assert vec(np.asarray([2]))[0] == 1 / 1.5  # True + 0.5 == 1.5


# ---- from test_review_round21 (git history: tests/test_review_round21.py)


def test_uint64_plus_nonnegative_int64_array_exact() -> None:
    # huge uint64 + non-negative signed values: modular uint64
    vec = vectorize(make_fn("    a = max(x, True)\n    return a + y", defaults="x, y"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64), np.asarray([0], dtype=np.int64))
    assert int(got[0]) == 9223372036854775809


def test_uint64_fit_plus_negative_int64_array_exact() -> None:
    # uint64 values that fit int64: int64 arithmetic handles negatives
    vec = vectorize(make_fn("    a = max(x, True)\n    return a + y", defaults="x, y"))
    got = vec(np.asarray([5], dtype=np.uint64), np.asarray([-3], dtype=np.int64))
    assert int(got[0]) == 2


def test_uint64_floordiv_signed_array_exact() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return a // y", defaults="x, y"))
    got = vec(np.asarray([7], dtype=np.uint64), np.asarray([-2], dtype=np.int64))
    assert int(got[0]) == -4


def test_uint64_mixed_sign_fallback_documented() -> None:
    # huge uint64 + negative signed: per-lane results fit no single
    # dtype; float64 approximation (documented divergence)
    vec = vectorize(make_fn("    a = max(x, True)\n    return a + y", defaults="x, y"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64), np.asarray([-1], dtype=np.int64))
    assert np.isclose(float(got[0]), 2**63)


# ---- from test_review_round22 (git history: tests/test_review_round22.py)


def test_uint64_mod_positive_int64_array_exact() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return a % y", defaults="x, y"))
    got = vec(np.asarray([2**63 + 3], dtype=np.uint64), np.asarray([2], dtype=np.int64))
    assert int(got[0]) == 1


def test_uint64_floordiv_positive_int64_array_exact() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return a // y", defaults="x, y"))
    got = vec(np.asarray([2**63 + 3], dtype=np.uint64), np.asarray([2], dtype=np.int64))
    assert int(got[0]) == 4611686018427387905


def test_nonnegative_signed_left_mod_uint64_exact() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return y % a", defaults="x, y"))
    got = vec(np.asarray([2**63 + 1], dtype=np.uint64), np.asarray([7], dtype=np.int64))
    assert int(got[0]) == 7


# ---- from test_review_round23 (git history: tests/test_review_round23.py)


def test_int64_minus_uint64_huge_exact() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return y - a", defaults="x, y"))
    got = vec(np.asarray([2**63], dtype=np.uint64), np.asarray([2**63 - 1], dtype=np.int64))
    assert int(got[0]) == -1


def test_sub_result_int64_min_exact() -> None:
    # -(2**63) is exactly int64-min: the mod-2**64 subtraction yields it
    vec = vectorize(make_fn("    a = max(x, True)\n    return y - a", defaults="x, y"))
    got = vec(np.asarray([2**63], dtype=np.uint64), np.asarray([0], dtype=np.int64))
    assert int(got[0]) == -(2**63)


def test_uint64_minus_signed_negative_result_exact() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return a - y", defaults="x, y"))
    got = vec(np.asarray([5], dtype=np.uint64), np.asarray([7], dtype=np.int64))
    assert int(got[0]) == -2


def test_sub_nonnegative_result_keeps_uint64() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return a - y", defaults="x, y"))
    got = vec(np.asarray([2**63 + 5], dtype=np.uint64), np.asarray([3], dtype=np.int64))
    assert int(got[0]) == 2**63 + 2


def test_sub_mixed_magnitude_fallback_documented() -> None:
    # per-lane differences spanning u64-only positives and int64
    # negatives fit no single dtype: float64 approximation
    vec = vectorize(make_fn("    a = max(x, True)\n    return a - y", defaults="x, y"))
    got = vec(np.asarray([2**63 + 5, 1], dtype=np.uint64), np.asarray([3, 4], dtype=np.int64))
    assert np.isclose(float(got[0]), 2**63 + 2)
    assert np.isclose(float(got[1]), -3)


# ---- from test_review_round24 (git history: tests/test_review_round24.py)


def test_literal_minus_uint64_exact() -> None:
    vec = vectorize(make_fn("    return 0 - max(x, True)"))
    got = vec(np.asarray([1], dtype=np.uint64))
    assert int(got[0]) == -1


def test_uint64_minus_uint64_exact() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return a - y", defaults="x, y"))
    got = vec(np.asarray([5], dtype=np.uint64), np.asarray([7], dtype=np.uint64))
    assert int(got[0]) == -2


def test_sub_nonnegative_result_keeps_uint64_unsigned() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return a - y", defaults="x, y"))
    got = vec(np.asarray([2**63 + 5], dtype=np.uint64), np.asarray([3], dtype=np.uint64))
    assert int(got[0]) == 2**63 + 2


def test_sub_mixed_magnitude_fallback_documented_unsigned() -> None:
    vec = vectorize(make_fn("    a = max(x, True)\n    return a - y", defaults="x, y"))
    got = vec(np.asarray([2**63 + 5, 1], dtype=np.uint64), np.asarray([3, 4], dtype=np.uint64))
    assert np.isclose(float(got[0]), 2**63 + 2)
    assert np.isclose(float(got[1]), -3)


# ---- from test_review_round25 (git history: tests/test_review_round25.py)


def test_uint64_minus_negative_literal_exact() -> None:
    vec = vectorize(make_fn("    return max(x, True) - -2"))
    got = vec(np.asarray([2**63 + 3], dtype=np.uint64))
    assert int(got[0]) == 2**63 + 5


def test_negative_literal_minus_uint64_exact() -> None:
    vec = vectorize(make_fn("    return -2 - max(x, True)"))
    got = vec(np.asarray([5], dtype=np.uint64))
    assert int(got[0]) == -7


def test_negative_literal_sub_int64_min_exact() -> None:
    # -(2**63 - 1) - True == -(2**63): exactly int64-min
    vec = vectorize(
        make_module(
            "import math\n\n"
            "BOUND = -(2**63 - 1)\n\n"
            "def subject(x, y=2.0):\n"
            "    return BOUND - max(x, True)\n"
        ).subject
    )
    got = vec(np.asarray([0], dtype=np.uint64))
    assert int(got[0]) == -(2**63)


def test_negative_literal_sub_past_int64_min_fallback() -> None:
    # -(2**63) - True == -(2**63) - 1: fits no dtype, documented float64
    vec = vectorize(
        make_module(
            "import math\n\n"
            "BOUND = -(2**63)\n\n"
            "def subject(x, y=2.0):\n"
            "    return BOUND - max(x, True)\n"
        ).subject
    )
    got = vec(np.asarray([0], dtype=np.uint64))
    assert np.isclose(float(got[0]), -(2**63) - 1)
