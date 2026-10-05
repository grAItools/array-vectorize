"""For-loops: phis, zero-trip semantics, kind mixing, nested and chained loops."""

from __future__ import annotations

import importlib.util
import itertools
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from support import just_vec, vec_of

from array_vectorize import VectorizationError, vectorize

_tmp = tempfile.TemporaryDirectory(prefix="vec_m3_")
_TMPDIR = Path(_tmp.name)
_seq = itertools.count()


def make_fn(body: str, extra: str = "") -> Callable[..., Any]:
    path = _TMPDIR / f"snippet_{next(_seq)}.py"
    path.write_text(
        "import math\n"
        "from math import exp\n"
        "GLOBAL_ARR = __import__('numpy').asarray([2.0, 4.0])\n"
        + extra
        + "\n\n\ndef subject(x, n=3):\n"
        + body
        + "\n"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.subject


# ---- from test_composition (git history: tests/test_composition.py)


def test_simple_accumulation() -> None:
    vec = vectorize(make_fn("    s = 0.0\n    for i in range(3):\n        s = s + x\n    return s"))
    got = vec(np.asarray([1.0, 2.0]))
    assert np.allclose(got, [3.0, 6.0])
    assert "for i in range(0, 3):" in vec.source
    assert "s_1 = s" in vec.source  # loop-carried phi


def test_loop_with_start_stop_step() -> None:
    vec = vectorize(
        make_fn("    s = 0.0\n    for i in range(1, 5, 2):\n        s = s + i * x\n    return s")
    )
    got = vec(np.asarray([1.0]))
    assert np.allclose(got, [(1 + 3) * 1.0])
    assert "for i in range(1, 5, 2):" in vec.source


def test_loop_negative_step() -> None:
    vec = vectorize(
        make_fn("    s = 0.0\n    for i in range(3, 0, -1):\n        s = s + i * x\n    return s")
    )
    assert np.allclose(vec(np.asarray([1.0])), [6.0])
    assert "for i in range(3, 0, -1):" in vec.source


def test_loop_index_used() -> None:
    vec = vectorize(
        make_fn("    s = 0.0\n    for i in range(4):\n        s = s + i\n    return s + x")
    )
    assert np.allclose(vec(np.asarray([0.0, 1.0])), [6.0, 7.0])


def test_loop_carried_two_vars() -> None:
    vec = vectorize(
        make_fn(
            "    a = 1.0\n"
            "    b = 0.0\n"
            "    for i in range(3):\n"
            "        b = b + x\n"
            "        a = a * 2.0\n"
            "    return a + b"
        )
    )
    got = vec(np.asarray([1.0]))
    assert np.allclose(got, [8.0 + 3.0])


def test_loop_local_var() -> None:
    vec = vectorize(make_fn("    for i in range(3):\n        t = i * x\n    return t + x"))
    assert np.allclose(vec(np.asarray([2.0])), [4.0 + 2.0])


def test_zero_trip_loop_matches_python() -> None:
    vec = vectorize(make_fn("    s = 5.0\n    for i in range(0):\n        s = s + x\n    return s"))
    assert np.allclose(vec(np.asarray([1.0])), [5.0])


def test_branch_inside_loop() -> None:
    vec = vectorize(
        make_fn(
            "    s = 0.0\n"
            "    for i in range(4):\n"
            "        if x > 0:\n"
            "            s = s + x\n"
            "        else:\n"
            "            s = s - 1.0\n"
            "    return s"
        )
    )
    xs = np.asarray([1.0, -1.0])
    expected = np.asarray([sum(x if x > 0 else -1.0 for _ in range(4)) for x in xs])
    assert np.allclose(vec(xs), expected)


def test_nested_loops() -> None:
    vec = vectorize(
        make_fn(
            "    s = 0.0\n"
            "    for i in range(3):\n"
            "        for j in range(2):\n"
            "            s = s + x\n"
            "    return s"
        )
    )
    assert np.allclose(vec(np.asarray([1.0])), [6.0])


def test_loop_var_shadow_param() -> None:
    vec = vectorize(
        make_fn("    n = 0.0\n    for n in range(3):\n        n2 = n\n    return n2 + x")
    )
    assert np.allclose(vec(np.asarray([1.0])), [2.0 + 1.0])


def test_loop_bound_closure_scalar() -> None:
    vec = vectorize(
        make_fn(
            "    s = 0.0\n    for i in range(N):\n        s = s + x\n    return s", extra="N = 4"
        )
    )
    assert np.allclose(vec(np.asarray([1.0])), [4.0])


def test_loop_bound_expression_folds() -> None:
    vec = vectorize(
        make_fn("    s = 0.0\n    for i in range(1 + 2):\n        s = s + x\n    return s")
    )
    assert "for i in range(0, 3):" in vec.source


def test_loop_bound_param_rejected() -> None:
    with pytest.raises(VectorizationError, match="constant ints"):
        vectorize(make_fn("    s = 0.0\n    for i in range(n):\n        s = s + x\n    return s"))


def test_loop_step_zero_rejected() -> None:
    with pytest.raises(VectorizationError, match="step cannot be zero"):
        vectorize(make_fn("    for i in range(3, 0, 0):\n        pass\n    return x"))


def test_return_inside_loop_rejected() -> None:
    with pytest.raises(VectorizationError, match="returns inside loop"):
        vectorize(
            make_fn(
                "    for i in range(3):\n        if x > 0:\n            return 1.0\n    return 0.0"
            )
        )


def test_assign_loop_var_rejected() -> None:
    with pytest.raises(VectorizationError, match="loop variable"):
        vectorize(make_fn("    for i in range(3):\n        i = i + 1\n    return x"))


def test_loop_over_closure_array() -> None:
    vec = vectorize(
        make_fn("    s = 0.0\n    for i in range(2):\n        s = s + x\n    return s + GLOBAL_ARR")
    )
    # element k: 2*x[k] + GLOBAL_ARR[k]
    assert np.allclose(vec(np.asarray([1.0, 1.0])), [4.0, 6.0])


def test_loop_dce_drops_dead_body_bindings() -> None:
    body = (
        "    s = 0.0\n"
        "    for i in range(3):\n"
        "        dead = x * 100.0\n"
        "        s = s + x\n"
        "    return s"
    )
    vec = vectorize(make_fn(body))
    # the original source in the docstring mentions 'dead' once; the code must not
    assert vec.source.count("dead") == 1


def test_loop_differential() -> None:
    fn = make_fn("    s = 1.0\n    for i in range(5):\n        s = s * (1.0 + x)\n    return s")
    vec = vectorize(fn)
    for x in (-0.2, 0.0, 0.3, 1.0):
        expected = 1.0
        for _ in range(5):
            expected *= 1.0 + x
        assert np.isclose(vec(np.asarray([x]))[0], expected)


# ---- from test_review_round1 (git history: tests/test_review_round1.py)


def test_loop_dce_keeps_cross_iteration_dependencies() -> None:
    body = (
        "    a = x\n"
        "    b = x\n"
        "    for i in range(3):\n"
        "        a = b\n"
        "        b = b + 1\n"
        "    return a"
    )
    vec, fn = vec_of(body)
    xs = np.asarray([1.0, 2.0])
    assert np.allclose(vec(xs), [fn(float(v)) for v in xs])


def test_empty_loop_body_emits_pass() -> None:
    vec = just_vec("    for i in range(3):\n        pass\n    return x + i")
    assert np.allclose(vec(np.asarray([1.0])), [3.0])


def test_zero_trip_loop_keeps_loop_var() -> None:
    vec = just_vec(
        "    i = x\n    a = x\n    for i in range(0):\n        a = a + 1\n    return i + a"
    )
    assert vec(np.asarray([1.0]))[0] == 2.0


# ---- from test_review_round2 (git history: tests/test_review_round2.py)


def test_loop_carried_kind_mixing() -> None:
    vec, fn = vec_of(
        "    b = x > 0\n"
        "    a = x\n"
        "    for i in range(2):\n"
        "        a = b + b\n"
        "        b = x + 0.5\n"
        "    return a"
    )
    xs = np.asarray([1.0, 2.0])
    assert np.allclose(vec(xs), [fn(float(v)) for v in xs])


def test_zero_trip_loop_kind_mixing() -> None:
    vec = vectorize(
        make_fn("    b = x > 0\n    for i in range(0):\n        b = x + 0.5\n    return b + b")
    )
    assert list(
        map(
            int,
            vectorize(
                make_fn(
                    "    b = x > 0\n    for i in range(0):\n        b = x + 0.5\n    return b + b"
                )
            )(np.asarray([1.0])),
        )
    ) == [2]
    del vec


def test_nested_loop_target_carried() -> None:
    vec = vectorize(
        make_fn(
            "    j = x\n"
            "    s = 0.0\n"
            "    for i in range(2):\n"
            "        s = s + j\n"
            "        for j in range(3):\n"
            "            pass\n"
            "    return s"
        )
    )
    assert vec(np.asarray([1.0]))[0] == 3.0


def test_nested_loop_same_var_name() -> None:
    vec = vectorize(
        make_fn(
            "    for i in range(2):\n        for i in range(3):\n            pass\n    return x + i"
        )
    )
    assert vec(np.asarray([1.0]))[0] == 3.0


# ---- from test_review_round3 (git history: tests/test_review_round3.py)


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


def test_range_two_args() -> None:
    vec = vectorize(
        make_fn("    s = x\n    for i in range(2, 5):\n        s = s + i\n    return s")
    )
    assert vec(np.asarray([0.0]))[0] == 9.0


# ---- from test_review_round4 (git history: tests/test_review_round4.py)


def test_chained_loop_kind_mixing() -> None:
    vec = vectorize(
        make_fn(
            "    a = 1\n"
            "    b = 1\n"
            "    c = 1\n"
            "    out = x\n"
            "    for i in range(4):\n"
            "        out = a + a\n"
            "        a = b\n"
            "        b = c\n"
            "        c = x > 0\n"
            "    return out"
        )
    )
    assert vec(np.asarray([1.0]))[0] == 2


def test_float_bool_loop_mix_keeps_float_values() -> None:
    # b is bool in iteration 1, float afterwards: the intify must not
    # truncate the float iterations (float64 for float-mixed names)
    vec = vectorize(
        make_fn(
            "    b = x > 0\n"
            "    a = x\n"
            "    for i in range(2):\n"
            "        a = b + b\n"
            "        b = x + 0.5\n"
            "    return a"
        )
    )
    assert np.allclose(vec(np.asarray([1.0, 2.0])), [3.0, 5.0])


# ---- from test_review_round5 (git history: tests/test_review_round5.py)


def test_float_mixed_propagates_through_assignment() -> None:
    vec = vectorize(
        make_fn(
            "    b = x\n    for i in range(1):\n        b = x + 0.5\n    c = b\n    return c + c"
        )
    )
    assert np.allclose(vec(np.asarray([1.0, 2.0])), [3.0, 5.0])


def test_true_division_loop_mix_keeps_float() -> None:
    vec = vectorize(
        make_fn(
            "    b = x > 0\n"
            "    a = x\n"
            "    for i in range(2):\n"
            "        a = b + b\n"
            "        b = x / 2.0\n"
            "    return a"
        )
    )
    assert vec(np.asarray([3.0]))[0] == 3.0


# ---- from test_review_round6 (git history: tests/test_review_round6.py)


def test_mixed_loop_operand_not_truncated_by_literal() -> None:
    vec = vectorize(
        make_fn("    b = x\n    for i in range(1):\n        b = x + 0.5\n    return b + 1")
    )
    assert np.allclose(vec(np.asarray([1.0, 2.0])), [2.5, 3.5])


def test_mixed_loop_operand_unary_negate() -> None:
    vec = vectorize(
        make_fn("    b = x\n    for i in range(1):\n        b = x + 0.5\n    return -b")
    )
    assert np.allclose(vec(np.asarray([1.0, 2.0])), [-1.5, -2.5])


def test_bool_loop_count_128() -> None:
    vec = vectorize(
        make_fn("    s = 0\n    for i in range(128):\n        s = s + (x > 0)\n    return s")
    )
    assert vec(np.asarray([1.0]))[0] == 128
