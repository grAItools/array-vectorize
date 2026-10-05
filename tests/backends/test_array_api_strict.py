"""array-api-strict backend twins of the behavior tests."""

from __future__ import annotations

import importlib.util
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from support import TMPDIR, make_fn, make_module

from array_vectorize import vectorize

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


# ---- from test_behavior (git history: tests/test_behavior.py)


def test_backend_array_api_strict() -> None:
    import array_api_strict as xps

    a = xps.asarray([1.0, -2.0, 3.0])
    got = vfn("relu")(a)
    assert isinstance(got, xps.asarray([1.0]).__class__)
    assert list(map(float, got)) == [1.0, 0.0, 3.0]


# ---- from test_review_round1 (git history: tests/test_review_round1.py)


def test_constant_scalar_math_on_strict_backend() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return x + math.sqrt(4.0)"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 3.0


# ---- from test_review_round2 (git history: tests/test_review_round2.py)


def test_literal_binding_math_call() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    y = 4\n    return x + math.sqrt(y)"))
    assert np.allclose(vec(np.asarray([1.0])), [3.0])
    assert float(vec(xps.asarray([1.0]))[0]) == 3.0


def test_scalar_default_math_call_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return x + math.sqrt(y)"))
    assert np.allclose(vec(np.asarray([1.0])), [1.0 + math.sqrt(2.0)])
    assert float(vec(xps.asarray([1.0]))[0]) == 1.0 + math.sqrt(2.0)


def test_loop_var_math_call_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(
        make_fn(
            "    s = 0.0\n    for i in range(3):\n        s = s + math.sin(i)\n    return s + x"
        )
    )
    got = vec(xps.asarray([0.0]))
    assert float(got[0]) == pytest.approx(math.sin(0) + math.sin(1) + math.sin(2))


def test_verify_on_strict_backend() -> None:
    import array_api_strict as xps

    path = TMPDIR / "identity_fn.py"
    path.write_text("def subject(x):\n    return x\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vec = vectorize(mod.subject, verify=(xps.asarray([1.0, 2.0]),))
    assert list(map(float, vec(xps.asarray([1.0, 2.0])))) == [1.0, 2.0]


def test_fallback_bool_arithmetic_on_strict_backend() -> None:
    import array_api_strict as xps

    fn = make_fn("    while x > 0:\n        x = x - 1\n    b = x > 0\n    return b + b")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vec = vectorize(fn, fallback=True)
    got = vec(xps.asarray([3.0]))
    assert list(map(int, got)) == [0]


# ---- from test_review_round3 (git history: tests/test_review_round3.py)


def test_min_max_literal_args_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return min(x, 3) + max(y, 1)"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 3.0  # min(1, 3) + max(2, 1) == 1 + 2


def test_bool_arithmetic_strict_backend() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return (x > 0) + (x > 0)"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 2.0


def test_bool_mixed_arithmetic_strict_backend() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return (x > 0) + 1"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 2.0


def test_computed_scalar_math_arg_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return x + math.sqrt(y + 0.0)"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 1.0 + math.sqrt(2.0)


def test_preamble_normalizes_scalar_params() -> None:
    import array_api_strict as xps

    # omitted defaults are raw Python scalars at runtime; the generated
    # preamble normalizes every parameter with xp.asarray
    vec = vectorize(make_fn("    return x + y"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 3.0


# ---- from test_review_round4 (git history: tests/test_review_round4.py)


def test_min_literal_strict_int_input() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return min(x, 3)"))
    got = vec(xps.asarray([1], dtype=xps.int64))
    assert int(got[0]) == 1


def test_min_literal_strict_float_default_param() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return min(y, 3)"))
    got = vec(xps.asarray([1.0]))
    # constant w.r.t. x: 0-d result is broadcast for comparison
    assert np.allclose(np.broadcast_to(np.asarray(got, dtype=float), (1,)), [2.0])


def test_int_default_scalar_promotion_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return x + y", defaults="x, y=1"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 2.0


def test_computed_local_scalar_math_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = 1.0 + 3.0\n    return x + math.sqrt(a)"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 3.0


def test_loop_index_math_expression_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(
        make_fn(
            "    s = 0.0\n    for i in range(2):\n        s = s + math.sqrt(i + 1.0)\n    return s"
        )
    )
    got = vec(xps.asarray([1.0]))
    # constant w.r.t. x: 0-d result is broadcast for comparison
    assert np.allclose(np.broadcast_to(np.asarray(got, dtype=float), (1,)), [1.0 + math.sqrt(2.0)])


def test_all_literal_minmax_refs_fold() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = 1.0\n    b = 2.0\n    return x + min(a, b)"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 2.0


# ---- from test_review_round5 (git history: tests/test_review_round5.py)


def test_bool_plus_float_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return (x > 0) + x"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 2.0


def test_bool_plus_float_literal_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return (x > 0) + 1.5"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 2.5


def test_scalar_phi_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(
        make_fn(
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


def test_minmax_promotion_matrix_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return min(x, 3)"))
    assert int(vec(xps.asarray([1], dtype=xps.int64))[0]) == 1

    vec = vectorize(make_fn("    return min(x, 1.5) + max(y, 1)"))
    got = vec(xps.asarray([2.0]))
    assert float(got[0]) == 3.5  # min(2, 1.5) + max(2, 1)

    vec = vectorize(make_fn("    return min(x, y)"))
    got = vec(xps.asarray([1.0, 3.0]), xps.asarray([2.0, 2.0]))
    assert np.allclose(np.asarray(got, dtype=float), [1.0, 2.0])


# ---- from test_review_round9 (git history: tests/test_review_round9.py)


def test_max_variadic_literals_strict() -> None:
    import array_api_strict as xps

    # 1 fits uint8 but 1.5 forces float64: BOTH literals must cast to
    # float64 (a uint8 literal would break strict same-dtype promotion)
    vec = vectorize(make_fn("    return max(x, 1, 1.5)"))
    got = vec(xps.asarray([2], dtype=xps.uint8))
    assert int(got[0]) == 2


# ---- from test_review_round11 (git history: tests/test_review_round11.py)


def test_bool_minmax_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return min(x > 0, x > 1)"))
    got = vec(xps.asarray([2, 0]))
    assert list(np.asarray(got, dtype=bool)) == [True, False]


def test_bool_minmax_strict_max() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return max(x > 0, x > 1)"))
    got = vec(xps.asarray([2, 0]))
    assert list(np.asarray(got, dtype=bool)) == [True, False]


# ---- from test_review_round12 (git history: tests/test_review_round12.py)


def test_bool_only_minmax_bitwise_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return min(x > 0, x > 1) & (x > 0)"))
    got = vec(xps.asarray([2]))
    assert bool(np.asarray(got).reshape(-1)[0])


# ---- from test_review_round13 (git history: tests/test_review_round13.py)


def test_minmax_runtime_bool_arithmetic_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = min(x, True)\n    return a + a"))
    got = vec(xps.asarray([True, False]))
    assert list(map(int, np.asarray(got))) == [2, 0]


# ---- from test_review_round14 (git history: tests/test_review_round14.py)


def test_unary_negate_maybe_bool_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = min(x, True)\n    return -a"))
    got = vec(xps.asarray([True, False]))
    assert list(map(int, np.asarray(got))) == [-1, 0]


# ---- from test_review_round15 (git history: tests/test_review_round15.py)


def test_bitwise_maybe_bool_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = min(x, True)\n    b = a & True\n    return b + b"))
    got = vec(xps.asarray([True, False]))
    assert list(map(int, np.asarray(got))) == [2, 0]


# ---- from test_review_round16 (git history: tests/test_review_round16.py)


def test_uint64_truediv_negative_divisor_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = max(x, True)\n    return a / -2"))
    got = vec(xps.asarray([3], dtype=xps.uint64))
    assert float(got[0]) == -1.5


# ---- from test_review_round17 (git history: tests/test_review_round17.py)


def test_truediv_bool_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return min(x, True) / 2"))
    got = vec(xps.asarray([True, False]))
    assert list(map(float, np.asarray(got))) == [0.5, 0.0]


# ---- from test_review_round18 (git history: tests/test_review_round18.py)


def test_truediv_bool_strict_after_restructure() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return min(x, True) / 2"))
    got = vec(xps.asarray([True, False]))
    assert list(map(float, np.asarray(got))) == [0.5, 0.0]


# ---- from test_review_round19 (git history: tests/test_review_round19.py)


def test_uint64_floordiv_negative_divisor_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return max(x, True) // -3"))
    got = vec(xps.asarray([5], dtype=xps.uint64))
    assert int(got[0]) == -2


def test_uint64_negative_literal_floordiv_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return -3 // max(x, True)"))
    got = vec(xps.asarray([5], dtype=xps.uint64))
    assert int(got[0]) == -1


# ---- from test_review_round20 (git history: tests/test_review_round20.py)


def test_uint64_mod_int64_min_divisor_strict() -> None:
    import array_api_strict as xps

    mod = make_module(
        "import math\n\n"
        "BOUND = -(2**63)\n\n"
        "def subject(x, y=2.0):\n"
        "    return max(x, True) % BOUND\n"
    )
    vec = vectorize(mod.subject)
    got = vec(xps.asarray([1], dtype=xps.uint64))
    assert int(got[0]) == -9223372036854775807


def test_arith_result_negative_power_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = (x > 0) + 1\n    return a ** -1"))
    assert float(vec(xps.asarray([2]))[0]) == 0.5


# ---- from test_review_round21 (git history: tests/test_review_round21.py)


def test_uint64_plus_nonnegative_int64_array_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = max(x, True)\n    return a + y", defaults="x, y"))
    got = vec(xps.asarray([2**63 + 1], dtype=xps.uint64), xps.asarray([0], dtype=xps.int64))
    assert int(got[0]) == 9223372036854775809


# ---- from test_review_round22 (git history: tests/test_review_round22.py)


def test_uint64_mod_positive_int64_array_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = max(x, True)\n    return a % y", defaults="x, y"))
    got = vec(xps.asarray([2**63 + 3], dtype=xps.uint64), xps.asarray([2], dtype=xps.int64))
    assert int(got[0]) == 1


def test_uint64_floordiv_positive_int64_array_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = max(x, True)\n    return a // y", defaults="x, y"))
    got = vec(xps.asarray([2**63 + 3], dtype=xps.uint64), xps.asarray([2], dtype=xps.int64))
    assert int(got[0]) == 4611686018427387905


# ---- from test_review_round23 (git history: tests/test_review_round23.py)


def test_int64_minus_uint64_huge_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = max(x, True)\n    return y - a", defaults="x, y"))
    got = vec(xps.asarray([2**63], dtype=xps.uint64), xps.asarray([2**63 - 1], dtype=xps.int64))
    assert int(got[0]) == -1


# ---- from test_review_round24 (git history: tests/test_review_round24.py)


def test_literal_minus_uint64_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return 0 - max(x, True)"))
    got = vec(xps.asarray([1], dtype=xps.uint64))
    assert int(got[0]) == -1


def test_uint64_minus_uint64_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = max(x, True)\n    return a - y", defaults="x, y"))
    got = vec(xps.asarray([5], dtype=xps.uint64), xps.asarray([7], dtype=xps.uint64))
    assert int(got[0]) == -2


# ---- from test_review_round25 (git history: tests/test_review_round25.py)


def test_uint64_minus_negative_literal_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return max(x, True) - -2"))
    got = vec(xps.asarray([2**63 + 3], dtype=xps.uint64))
    assert int(got[0]) == 2**63 + 5


def test_negative_literal_minus_uint64_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return -2 - max(x, True)"))
    got = vec(xps.asarray([5], dtype=xps.uint64))
    assert int(got[0]) == -7
