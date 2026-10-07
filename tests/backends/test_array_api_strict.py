"""array-api-strict backend twins of the behavior tests."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import support

import array_vectorize

# ----------------------------------------------- corpus and parameter preamble


def test_backend_array_api_strict() -> None:
    import array_api_strict as xps

    a = xps.asarray([1.0, -2.0, 3.0])
    got = support.vfn("relu")(a)
    assert isinstance(got, xps.asarray([1.0]).__class__)
    assert list(map(float, got)) == [1.0, 0.0, 3.0]


def test_preamble_normalizes_scalar_params() -> None:
    import array_api_strict as xps

    # omitted defaults are raw Python scalars at runtime; the generated
    # preamble normalizes every parameter with xp.asarray
    vec = array_vectorize.vectorize(support.make_fn("    return x + y"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 3.0


def test_int_default_scalar_promotion_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    return x + y", defaults="x, y=1"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 2.0


# ------------------------------------------------------- scalar math arguments


def test_constant_scalar_math_on_strict_backend() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    return x + math.sqrt(4.0)"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 3.0


def test_literal_binding_math_call() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    y = 4\n    return x + math.sqrt(y)"))
    assert np.allclose(vec(np.asarray([1.0])), [3.0])
    assert float(vec(xps.asarray([1.0]))[0]) == 3.0


def test_scalar_default_math_call_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    return x + math.sqrt(y)"))
    assert np.allclose(vec(np.asarray([1.0])), [1.0 + math.sqrt(2.0)])
    assert float(vec(xps.asarray([1.0]))[0]) == 1.0 + math.sqrt(2.0)


def test_computed_scalar_math_arg_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    return x + math.sqrt(y + 0.0)"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 1.0 + math.sqrt(2.0)


def test_computed_local_scalar_math_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(
        support.make_fn("    a = 1.0 + 3.0\n    return x + math.sqrt(a)")
    )
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 3.0


def test_loop_var_math_call_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(
        support.make_fn(
            "    s = 0.0\n    for i in range(3):\n        s = s + math.sin(i)\n    return s + x"
        )
    )
    got = vec(xps.asarray([0.0]))
    assert float(got[0]) == pytest.approx(math.sin(0) + math.sin(1) + math.sin(2))


def test_loop_index_math_expression_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(
        support.make_fn(
            "    s = 0.0\n    for i in range(2):\n        s = s + math.sqrt(i + 1.0)\n    return s"
        )
    )
    got = vec(xps.asarray([1.0]))
    # constant w.r.t. x: 0-d result is broadcast for comparison
    assert np.allclose(np.broadcast_to(np.asarray(got, dtype=float), (1,)), [1.0 + math.sqrt(2.0)])


def test_scalar_phi_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(
        support.make_fn(
            "    b = 4.0\n"
            "    for i in range(0):\n"
            "        if x > 0:\n"
            "            b = 9.0\n"
            "    return math.sqrt(b)"
        )
    )
    got = vec(xps.asarray([1.0]))
    # constant w.r.t. x: 0-d result is broadcast for comparison
    assert np.allclose(np.broadcast_to(np.asarray(got, dtype=float), (1,)), [2.0])


# --------------------------------------------------------- verify and fallback


def test_verify_on_strict_backend() -> None:
    import array_api_strict as xps

    mod = support.make_module("def subject(x):\n    return x\n")
    vec = array_vectorize.vectorize(mod.subject, verify=(xps.asarray([1.0, 2.0]),))
    assert list(map(float, vec(xps.asarray([1.0, 2.0])))) == [1.0, 2.0]


def test_fallback_bool_arithmetic_on_strict_backend() -> None:
    import array_api_strict as xps

    fn = support.make_fn("    while x > 0:\n        x = x - 1\n    b = x > 0\n    return b + b")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vec = array_vectorize.vectorize(fn, fallback=True)
    got = vec(xps.asarray([3.0]))
    assert list(map(int, got)) == [0]


# ----------------------------------------------------- bool arithmetic and pow


def test_bool_arithmetic_strict_backend() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    return (x > 0) + (x > 0)"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 2.0


def test_bool_mixed_arithmetic_strict_backend() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    return (x > 0) + 1"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 2.0


def test_bool_plus_float_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    return (x > 0) + x"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 2.0


def test_bool_plus_float_literal_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    return (x > 0) + 1.5"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 2.5


def test_arith_result_negative_power_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    a = (x > 0) + 1\n    return a ** -1"))
    assert float(vec(xps.asarray([2]))[0]) == 0.5


# ----------------------------------------------------------- min/max promotion


def test_min_max_literal_args_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    return min(x, 3) + max(y, 1)"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 3.0  # min(1, 3) + max(2, 1) == 1 + 2


def test_min_literal_strict_int_input() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    return min(x, 3)"))
    got = vec(xps.asarray([1], dtype=xps.int64))
    assert int(got[0]) == 1


def test_min_literal_strict_float_default_param() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    return min(y, 3)"))
    got = vec(xps.asarray([1.0]))
    # constant w.r.t. x: 0-d result is broadcast for comparison
    assert np.allclose(np.broadcast_to(np.asarray(got, dtype=float), (1,)), [2.0])


def test_all_literal_minmax_refs_fold() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(
        support.make_fn("    a = 1.0\n    b = 2.0\n    return x + min(a, b)")
    )
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 2.0


def test_minmax_promotion_matrix_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    return min(x, 3)"))
    assert int(vec(xps.asarray([1], dtype=xps.int64))[0]) == 1

    vec = array_vectorize.vectorize(support.make_fn("    return min(x, 1.5) + max(y, 1)"))
    got = vec(xps.asarray([2.0]))
    assert float(got[0]) == 3.5  # min(2, 1.5) + max(2, 1)

    vec = array_vectorize.vectorize(support.make_fn("    return min(x, y)"))
    got = vec(xps.asarray([1.0, 3.0]), xps.asarray([2.0, 2.0]))
    assert np.allclose(np.asarray(got, dtype=float), [1.0, 2.0])


def test_max_variadic_literals_strict() -> None:
    import array_api_strict as xps

    # 1 fits uint8 but 1.5 forces float64: BOTH literals must cast to
    # float64 (a uint8 literal would break strict same-dtype promotion)
    vec = array_vectorize.vectorize(support.make_fn("    return max(x, 1, 1.5)"))
    got = vec(xps.asarray([2], dtype=xps.uint8))
    assert int(got[0]) == 2


def test_bool_minmax_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    return min(x > 0, x > 1)"))
    got = vec(xps.asarray([2, 0]))
    assert list(np.asarray(got, dtype=bool)) == [True, False]


def test_bool_minmax_strict_max() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    return max(x > 0, x > 1)"))
    got = vec(xps.asarray([2, 0]))
    assert list(np.asarray(got, dtype=bool)) == [True, False]


def test_bool_only_minmax_bitwise_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    return min(x > 0, x > 1) & (x > 0)"))
    got = vec(xps.asarray([2]))
    assert bool(np.asarray(got).reshape(-1)[0])


# ----------------------------------- min/max: bool dtype known only at runtime


def test_minmax_runtime_bool_arithmetic_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    a = min(x, True)\n    return a + a"))
    got = vec(xps.asarray([True, False]))
    assert list(map(int, np.asarray(got))) == [2, 0]


def test_unary_negate_maybe_bool_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    a = min(x, True)\n    return -a"))
    got = vec(xps.asarray([True, False]))
    assert list(map(int, np.asarray(got))) == [-1, 0]


def test_bitwise_maybe_bool_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(
        support.make_fn("    a = min(x, True)\n    b = a & True\n    return b + b")
    )
    got = vec(xps.asarray([True, False]))
    assert list(map(int, np.asarray(got))) == [2, 0]


def test_truediv_bool_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    return min(x, True) / 2"))
    got = vec(xps.asarray([True, False]))
    assert list(map(float, np.asarray(got))) == [0.5, 0.0]


# -------------------------------------------------------------- uint64 add/sub


def test_uint64_plus_nonnegative_int64_array_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(
        support.make_fn("    a = max(x, True)\n    return a + y", defaults="x, y")
    )
    got = vec(xps.asarray([2**63 + 1], dtype=xps.uint64), xps.asarray([0], dtype=xps.int64))
    assert int(got[0]) == 9223372036854775809


def test_uint64_minus_uint64_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(
        support.make_fn("    a = max(x, True)\n    return a - y", defaults="x, y")
    )
    got = vec(xps.asarray([5], dtype=xps.uint64), xps.asarray([7], dtype=xps.uint64))
    assert int(got[0]) == -2


def test_int64_minus_uint64_huge_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(
        support.make_fn("    a = max(x, True)\n    return y - a", defaults="x, y")
    )
    got = vec(xps.asarray([2**63], dtype=xps.uint64), xps.asarray([2**63 - 1], dtype=xps.int64))
    assert int(got[0]) == -1


def test_literal_minus_uint64_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    return 0 - max(x, True)"))
    got = vec(xps.asarray([1], dtype=xps.uint64))
    assert int(got[0]) == -1


def test_uint64_minus_negative_literal_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    return max(x, True) - -2"))
    got = vec(xps.asarray([2**63 + 3], dtype=xps.uint64))
    assert int(got[0]) == 2**63 + 5


def test_negative_literal_minus_uint64_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    return -2 - max(x, True)"))
    got = vec(xps.asarray([5], dtype=xps.uint64))
    assert int(got[0]) == -7


# ------------------------------------------------- uint64 floordiv/mod/truediv


def test_uint64_floordiv_negative_divisor_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    return max(x, True) // -3"))
    got = vec(xps.asarray([5], dtype=xps.uint64))
    assert int(got[0]) == -2


def test_uint64_negative_literal_floordiv_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    return -3 // max(x, True)"))
    got = vec(xps.asarray([5], dtype=xps.uint64))
    assert int(got[0]) == -1


def test_uint64_floordiv_positive_int64_array_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(
        support.make_fn("    a = max(x, True)\n    return a // y", defaults="x, y")
    )
    got = vec(xps.asarray([2**63 + 3], dtype=xps.uint64), xps.asarray([2], dtype=xps.int64))
    assert int(got[0]) == 4611686018427387905


def test_uint64_mod_positive_int64_array_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(
        support.make_fn("    a = max(x, True)\n    return a % y", defaults="x, y")
    )
    got = vec(xps.asarray([2**63 + 3], dtype=xps.uint64), xps.asarray([2], dtype=xps.int64))
    assert int(got[0]) == 1


def test_uint64_mod_int64_min_divisor_strict() -> None:
    import array_api_strict as xps

    mod = support.make_module(
        "import math\n\n"
        "BOUND = -(2**63)\n\n"
        "def subject(x, y=2.0):\n"
        "    return max(x, True) % BOUND\n"
    )
    vec = array_vectorize.vectorize(mod.subject)
    got = vec(xps.asarray([1], dtype=xps.uint64))
    assert int(got[0]) == -9223372036854775807


def test_uint64_truediv_negative_divisor_strict() -> None:
    import array_api_strict as xps

    vec = array_vectorize.vectorize(support.make_fn("    a = max(x, True)\n    return a / -2"))
    got = vec(xps.asarray([3], dtype=xps.uint64))
    assert float(got[0]) == -1.5


@pytest.mark.parametrize("op", ["%", "//"])
@pytest.mark.parametrize("bound", [False, True])
def test_negative_integer_with_unsigned_input(op: str, bound: bool) -> None:
    import array_api_strict as xps

    body = f"    return -7 {op} x"
    if bound:
        body = f"    n = -7\n    return n {op} x"
    scalar = support.make_fn(body)
    values = [1, 2, 2**63, 2**64 - 1]
    got = array_vectorize.vectorize(scalar)(xps.asarray(values, dtype=xps.uint64))
    assert [int(v) for v in got] == [scalar(v) for v in values]
    assert np.asarray(got).dtype.kind == ("u" if op == "%" else "i")


@pytest.mark.parametrize("dtype", ["bool", "int64", "uint64"])
def test_true_division_of_integer_inputs(dtype: str) -> None:
    import array_api_strict as xps

    scalar = support.make_fn("    return x / y", defaults="x, y")
    values = [True, False] if dtype == "bool" else [1, 2, 3]
    if dtype == "uint64":
        values += [2**63, 2**64 - 1]
    denominator = True if dtype == "bool" else 3
    got = array_vectorize.vectorize(scalar)(
        xps.asarray(values, dtype=getattr(xps, dtype)),
        xps.asarray([denominator] * len(values), dtype=getattr(xps, dtype)),
    )
    assert [float(v) for v in got] == [scalar(v, denominator) for v in values]


@pytest.mark.parametrize("assignment", ["y = y + 1", "y += 1", "y = y + y"])
@pytest.mark.parametrize("trips", [0, 1, 3])
def test_boolean_input_carried_through_loop(assignment: str, trips: int) -> None:
    import array_api_strict as xps

    scalar = support.make_fn(
        f"    y = x\n    for i in range({trips}):\n        {assignment}\n    return y"
    )
    got = array_vectorize.vectorize(scalar)(xps.asarray([False, True]))
    assert [int(v) for v in got] == [scalar(False), scalar(True)]
    assert np.asarray(got).dtype.kind == ("b" if trips == 0 else "i")


@pytest.mark.parametrize(
    "condition", ["i == 0", "i == 0 and i < 2", "i == 0 or i < 0", "not (i != 0)", "i + 1 == 1"]
)
@pytest.mark.parametrize("statement", [False, True])
@pytest.mark.parametrize("trips", [1, 3])
def test_loop_index_selects_branch_without_dtype_promotion(
    condition: str, statement: bool, trips: int
) -> None:
    import array_api_strict as xps

    body = f"        y = x if {condition} else y + 1"
    if statement:
        body = f"        if {condition}:\n            y = x\n        else:\n            y = y + 1"
    scalar = support.make_fn(f"    y = True\n    for i in range({trips}):\n{body}\n    return y")
    values = [-0.0, 0.0, -1.0, 1.0, np.nextafter(0.0, 1.0), -np.inf, np.inf, np.nan]
    got = array_vectorize.vectorize(scalar)(xps.asarray(values, dtype=xps.float64))
    np.testing.assert_equal(np.asarray(got), [scalar(v) for v in values])
    assert np.asarray(got).dtype.kind == "f"
    if trips == 1:
        np.testing.assert_equal(np.signbit(np.asarray(got)), np.signbit(values))


def test_integer_division_does_not_round_the_denominator_first() -> None:
    import array_api_strict as xps

    scalar = support.make_fn("    return x / y", defaults="x, y")
    left = [1, -1, -(2**63), 2**63 - 1, 0]
    right = [2**53 + 1, 2**53 + 1, -1, 3, -1]
    got = array_vectorize.vectorize(scalar)(
        xps.asarray(left, dtype=xps.int64),
        xps.asarray(right, dtype=xps.int64),
    )
    expected = [a / b for a, b in zip(left, right, strict=True)]
    np.testing.assert_array_equal(np.asarray(got), expected)
    np.testing.assert_array_equal(np.signbit(np.asarray(got)), np.signbit(expected))


def test_unsigned_floor_division_preserves_unrepresentable_negative_sign() -> None:
    import array_api_strict as xps

    scalar = support.make_fn("    return x // -1")
    values = [0, 1, 2**63, 2**63 + 1, 2**64 - 1]
    got = array_vectorize.vectorize(scalar)(xps.asarray(values, dtype=xps.uint64))
    # No integer array dtype holds these results; the lattice uses float64.
    assert np.asarray(got).dtype.kind == "f"
    np.testing.assert_array_equal(np.asarray(got), [float(-v) for v in values])
    np.testing.assert_array_equal(np.signbit(np.asarray(got)), [v != 0 for v in values])
