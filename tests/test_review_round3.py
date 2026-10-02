"""Regression tests for the adversarial-review findings (round 3)."""

from __future__ import annotations

import importlib.util
import itertools
import math
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from vectorizer import vectorize

_tmp = tempfile.TemporaryDirectory(prefix="vec_rev3_")
_TMPDIR = Path(_tmp.name)
_seq = itertools.count()


def make_fn(body: str, extra: str = "") -> Callable[..., Any]:
    path = _TMPDIR / f"snippet_{next(_seq)}.py"
    path.write_text(
        "import math\nimport numpy as np\n" + extra + "\n\n\ndef subject(x, y=2.0):\n" + body + "\n"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.subject


# 1: literal facts do not survive loop boundaries


def test_zero_trip_loop_literal_not_substituted() -> None:
    vec = vectorize(
        make_fn("    a = x\n    for i in range(0):\n        a = 4.0\n    return math.sqrt(a)")
    )
    assert np.allclose(vec(np.asarray([9.0, 16.0])), [3.0, 4.0])


def test_executed_loop_literal_not_substituted() -> None:
    vec = vectorize(
        make_fn("    a = 4.0\n    for i in range(1):\n        a = 9.0\n    return math.sqrt(a)")
    )
    assert np.allclose(vec(np.asarray([9.0])), [3.0])


def test_conditional_loop_literal_not_substituted() -> None:
    vec = vectorize(
        make_fn(
            "    a = 4.0\n"
            "    for i in range(1):\n"
            "        if x > 0:\n"
            "            a = 9.0\n"
            "    return math.sqrt(a)"
        )
    )
    # zero-trip semantics per input are unified by the loop machinery: the
    # phi feeds 4.0 before the first iteration
    assert np.allclose(vec(np.asarray([1.0])), [3.0])


# 2: numeric-to-boolean loop transitions


def test_int_to_bool_loop_transition() -> None:
    vec = vectorize(
        make_fn(
            "    b = 1\n"
            "    a = x\n"
            "    for i in range(2):\n"
            "        a = b + b\n"
            "        b = x > 0\n"
            "    return a"
        )
    )
    assert vectorize(make_fn("    return 0")) is not None
    assert vec(np.asarray([1.0]))[0] == 2


def test_float_to_bool_loop_transition() -> None:
    vec = vectorize(
        make_fn(
            "    b = 1.5\n"
            "    a = x\n"
            "    for i in range(2):\n"
            "        a = b + b\n"
            "        b = x > 0\n"
            "    return a"
        )
    )
    got = vec(np.asarray([1.0]))
    assert got[0] == 2.0  # iteration 2: True + True == 2


# 3: polymorphic builtins keep exact integers


def test_abs_exact_large_int() -> None:
    vec = vectorize(make_fn("    return x + abs(9007199254740993)"))
    got = vec(np.asarray([0], dtype=np.int64))
    assert got[0] == 9007199254740993


def test_abs_int_bitwise() -> None:
    vec = vectorize(make_fn("    return abs(3) & x"))
    got = vec(np.asarray([1], dtype=np.int64))
    assert got[0] == 1


def test_min_max_literal_args_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return min(x, 3) + max(y, 1)"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 3.0  # min(1, 3) + max(2, 1) == 1 + 2


# 4: lambda matching distinguishes signed zeros and mixed numeric types


def test_lambda_signed_zero_constants() -> None:
    path = _TMPDIR / "lam_zeros.py"
    path.write_text(
        "import math\n"
        "f1, f2 = (\n"
        "    lambda x: math.copysign(x, 0.0),\n"
        "    lambda x: math.copysign(x, -0.0),\n"
        ")\n"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert vectorize(mod.f2)(np.asarray([1.0]))[0] == -1.0
    assert vectorize(mod.f1)(np.asarray([1.0]))[0] == 1.0


def test_lambda_int_vs_bool_vs_float_constants() -> None:
    path = _TMPDIR / "lam_types.py"
    path.write_text("g1, g2, g3 = (lambda x: x + 1), (lambda x: x + True), (lambda x: x + 1.0)\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert vectorize(mod.g2)(np.asarray([1.0]))[0] == 2.0
    assert vectorize(mod.g3)(np.asarray([1.0]))[0] == 2.0


# 5: protect_domains never creates forward or self references


def test_protect_domains_self_reference_guard() -> None:
    vec = vectorize(
        make_fn(
            "    if x < 0:\n"
            "        return 0.0\n"
            "    y = math.sqrt(x)\n"
            "    return y if y > 1 else 0.0"
        ),
        protect_domains=True,
    )
    with np.errstate(all="raise"):
        got = vec(np.asarray([-1.0, 4.0]))
    assert np.allclose(got, [0.0, 2.0])


def test_protect_domains_forward_reference_guard() -> None:
    # the liveness condition references z, which is bound after the sqrt:
    # the clamp must be skipped (not crash), values stay exact
    vec = vectorize(
        make_fn(
            "    y = math.sqrt(x)\n"
            "    if x > 0:\n"
            "        z = x\n"
            "    else:\n"
            "        z = 1.0\n"
            "    return y if z > 0 else 0.0"
        ),
        protect_domains=True,
    )
    with np.errstate(all="raise"):
        got = vec(np.asarray([0.0, 4.0]))
    assert np.allclose(got, [0.0, 2.0])


# 6: boolean arithmetic on strict backends


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


# 7: computed scalar math arguments


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


# 8: scalar arguments to casts


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


# 9: mypy strict passes (covered by make check); smoke the pipeline end to end


def test_round3_pipeline_smoke() -> None:
    vec = vectorize(
        make_fn(
            "    a = x\n    for i in range(2):\n        a = a + math.sqrt(abs(x) + 1)\n    return a"
        ),
        protect_domains=True,
    )
    xs = np.asarray([0.5, 2.0])
    want = xs
    v = xs
    for _ in range(2):
        v = v + np.sqrt(np.abs(xs) + 1)
    del want
    assert np.allclose(vec(xs), v)


# round-3 follow-ups: intify / sanitize / pow branches


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


def test_range_two_args() -> None:
    vec = vectorize(
        make_fn("    s = x\n    for i in range(2, 5):\n        s = s + i\n    return s")
    )
    assert vec(np.asarray([0.0]))[0] == 9.0
