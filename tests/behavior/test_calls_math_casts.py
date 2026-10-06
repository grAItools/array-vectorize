"""Math and builtin calls, casts, pow, floor-mod, abs/round, scalar-argument math."""

from __future__ import annotations

import numpy as np
from support import make_fn, vfn

from array_vectorize import vectorize

X = np.asarray([-2.0, -0.5, 0.0, 0.5, 2.0, 10.0])
Y = np.asarray([3.0, 1.5, 1.0, 0.5, -1.0, -3.0])


# ------------------------------------------------------------------ math calls


def test_math_calls() -> None:
    mx = X * (X > 0)  # keep sqrt domain valid
    expected = np.sqrt(mx) + np.exp(mx) + np.log1p(mx) + np.sin(mx)
    got = vfn("math_calls")(mx)
    with np.errstate(invalid="ignore"):
        assert np.allclose(got, expected, equal_nan=True)


def test_math_two_arg() -> None:
    expected = np.arctan2(X, Y) + np.hypot(X, Y) + np.copysign(X, Y)
    assert np.allclose(vfn("math_two_arg")(X, Y), expected)


def test_complex_expr() -> None:
    t = np.exp(-(X * X + Y * Y))
    expected = t / (1.0 + t)
    assert np.allclose(vfn("complex_expr")(X, Y), expected)


# -------------------------------------------------- pow and floor-division/mod


def test_pow_op() -> None:
    expected = np.power(X, 2) + np.power(2.0, X)
    assert np.allclose(vfn("pow_op")(X), expected)


def test_floor_mod_ops() -> None:
    expected = np.floor_divide(X, 2) + np.remainder(X, 3)
    got = vfn("floor_mod_ops")(X)
    assert np.allclose(got, expected)


def test_div_mod_sign_semantics_match_python() -> None:
    fn_mod = make_fn("    return x % 3")
    fn_floordiv = make_fn("    return x // 3")
    xs = np.asarray([-7.0, -6.5, 6.5, 7.0])
    assert np.allclose(vectorize(fn_mod)(xs), [x % 3 for x in xs])
    assert np.allclose(vectorize(fn_floordiv)(xs), [x // 3 for x in xs])


# --------------------------------------------------------------- abs and round


def test_abs_round() -> None:
    assert np.allclose(vfn("abs_round")(X), np.abs(X) + np.round(X))


def test_round_half_even() -> None:
    fn = make_fn("    return round(x)")
    xs = np.asarray([0.5, 1.5, 2.5, -0.5, -1.5])
    assert np.array_equal(vectorize(fn)(xs), [round(x) for x in xs])  # banker's rounding


def test_abs_round_loop_var() -> None:
    vec = vectorize(
        make_fn(
            "    s = x\n"
            "    for i in range(3):\n"
            "        s = s + abs(i - 1) + round(i / 2)\n"
            "    return s"
        )
    )
    want = 0.0
    for i in range(3):
        want += abs(i - 1) + round(i / 2)
    assert vec(np.asarray([0.0]))[0] == want


def test_abs_exact_large_int() -> None:
    vec = vectorize(make_fn("    return x + abs(9007199254740993)"))
    got = vec(np.asarray([0], dtype=np.int64))
    assert got[0] == 9007199254740993


def test_abs_int_bitwise() -> None:
    vec = vectorize(make_fn("    return abs(3) & x"))
    got = vec(np.asarray([1], dtype=np.int64))
    assert got[0] == 1


# ----------------------------------------------------------------------- casts


def test_casts() -> None:
    got = vfn("casts")(X)
    expected = X.astype(np.int64) + X + X.astype(bool) + X.astype(np.int64)
    assert np.allclose(got, expected)


def test_cast_scalar_default_param() -> None:
    vec = vectorize(make_fn("    return x + int(y)"))
    got = vec(np.asarray([1.0]))
    assert got[0] == 3.0  # y defaults to 2.0 -> int(2.0) == 2


def test_cast_literal_folds() -> None:
    vec = vectorize(make_fn("    return x + int(3.0)"))
    got = vec(np.asarray([1.0]))
    assert got[0] == 4.0


def test_trunc_scalar_default_param() -> None:
    vec = vectorize(make_fn("    return x + math.trunc(y)"))
    got = vec(np.asarray([1.0]))
    assert got[0] == 3.0  # math.trunc(2.0) == 2
