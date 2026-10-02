"""Regression tests for the adversarial-review findings (round 1)."""

from __future__ import annotations

import importlib.util
import itertools
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vectorizer import VectorizationError, vectorize
from vectorizer._verify import verify_match

_tmp = tempfile.TemporaryDirectory(prefix="vec_rev1_")
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


def vec_of(body: str, extra: str = "") -> tuple[Callable[..., Any], Callable[..., Any]]:
    fn = make_fn(body, extra)
    return vectorize(fn), fn


def just_vec(body: str, extra: str = "") -> Callable[..., Any]:
    return vectorize(make_fn(body, extra))


# 1: loop DCE must keep loop-carried feedback alive


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


# 2: CSE temps must not collide with lowered binding names


def test_cse_temp_does_not_collide_with_ssa_bindings() -> None:
    vec, fn = vec_of("    t = x + 1\n    t = t + 1\n    return t + math.sin(x) + math.sin(x)")
    xs = np.asarray([1.0])
    assert np.isclose(vec(xs)[0], fn(1.0))


# 3: signed zeros are distinct for CSE


def test_signed_zero_not_cse_merged() -> None:
    vec = just_vec("    return math.copysign(x, 0.0) + math.copysign(x, -0.0)")
    assert np.allclose(vec(np.asarray([1.0])), [0.0])


# 4: helper calls with keywords are rejected, not silently altered


def test_helper_keyword_call_rejected() -> None:
    fn = make_fn("    return helper(x, y=9)", extra="def helper(x, y=2):\n    return x + y\n")
    with pytest.raises(VectorizationError, match="keyword"):
        vectorize(fn)


# 5: same-named helper cannot shadow the caller


def test_same_named_helper_does_not_shadow_caller() -> None:
    fn = make_fn(
        "    return helper(x) * 10",
        extra=(
            "def make_helper():\n"
            "    def subject(x):\n"
            "        return x + 1\n"
            "    return subject\n"
            "helper = make_helper()\n"
        ),
    )
    assert vectorize(fn)(np.asarray([1.0]))[0] == 20.0


# 6: boolean arithmetic is exact through assignments and aug-assign


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


# 7: protect_domains never changes results (incl. -0.0) and never warns


def test_protect_domains_preserves_signed_zero() -> None:
    vec = vectorize(
        make_fn("    return math.copysign(1.0, math.sqrt(x)) if x <= 0 else 5.0"),
        protect_domains=True,
    )
    assert vec(np.asarray([-0.0]))[0] == -1.0


def test_protect_domains_no_warning_on_poles() -> None:
    vec = vectorize(make_fn("    return math.log(x) if x > 0 else 0.0"), protect_domains=True)
    with np.errstate(all="raise"):
        got = vec(np.asarray([-1.0, 1.0]))
    assert np.allclose(got, [0.0, 0.0])


# 8: multiple lambdas on one line


def test_multiple_lambdas_on_one_line() -> None:
    path = _TMPDIR / "multi_lambda.py"
    path.write_text("f1, f2 = (lambda x: x + 1), (lambda x: x + 2)\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert vectorize(mod.f2)(np.asarray([1.0]))[0] == 3.0
    assert vectorize(mod.f1)(np.asarray([1.0]))[0] == 2.0


# 9/10/11: verification oracle


def test_verify_detects_bool_miscompile() -> None:
    with pytest.raises(VectorizationError, match="verification failed"):
        verify_match(
            lambda x: np.array([False, True]),  # wrong: should be [0, 2]
            lambda x: 2 if x > 0 else 0,
            (np.array([-1.0, 1.0]),),
        )


def test_verify_accepts_matching_infinities() -> None:
    verify_match(lambda x: x, lambda x: x, (np.array([np.inf, -np.inf]),))


def test_verify_rejects_inf_vs_finite() -> None:
    with pytest.raises(VectorizationError, match="verification failed"):
        verify_match(lambda x: np.array([0.0]), lambda x: float("inf"), (np.array([1.0]),))


def test_verify_passes_scalar_arguments_through() -> None:
    fn = make_fn("    return x + y")
    vec = vectorize(fn, verify=(np.asarray([1.0]), 3))
    assert vec(np.asarray([1.0]), 3)[0] == 4.0


def test_verify_detects_wrong_constant_output() -> None:
    # a "vectorized" function returning the wrong constant must be caught
    with pytest.raises(VectorizationError, match="verification failed"):
        verify_match(lambda x: np.array([1.0]), lambda x: x * 5.0, (np.array([2.0]),))


# 12: fallback handles array kwargs and keyword-only array calls


def test_fallback_array_kwargs() -> None:
    fn = make_fn("    while x < y:\n        x = x + 1\n    return x")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = vectorize(fn, fallback=True)
    assert np.allclose(v(x=np.asarray([1.0, 3.0])), [2.0, 3.0])
    assert np.allclose(v(np.asarray([1.0, 3.0]), y=np.asarray([2.0, 4.0])), [2.0, 4.0])


# 13: closure arrays keep the function's own kwonly defaults


def test_hidden_params_preserve_kwonly_defaults() -> None:
    path = _TMPDIR / "kwonly_closure.py"
    path.write_text(
        "import numpy as np\n"
        "C = np.asarray([2.0])\n"
        "\n"
        "\n"
        "def subject(x, *, scale=3):\n"
        "    return x * scale + C\n"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vec = vectorize(mod.subject)
    assert np.allclose(vec(np.asarray([1.0])), [5.0])
    assert np.allclose(vec(np.asarray([1.0]), scale=4), [6.0])


# 14: a parameter named xp stays callable by keyword


def test_reserved_xp_param_keyword_call() -> None:
    path = _TMPDIR / "xp_param.py"
    path.write_text("import numpy as np\n\n\ndef subject(xp):\n    return xp + 1\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vec = vectorize(mod.subject)
    assert np.allclose(vec(xp=np.asarray([1.0])), [2.0])
    assert "def subject_vec(xp):" in vec.source


# 15: retained loops with empty bodies generate valid Python


def test_empty_loop_body_emits_pass() -> None:
    vec = just_vec("    for i in range(3):\n        pass\n    return x + i")
    assert np.allclose(vec(np.asarray([1.0])), [3.0])


# 16: zero-trip loops keep the pre-loop loop-variable value


def test_zero_trip_loop_keeps_loop_var() -> None:
    vec = just_vec(
        "    i = x\n    a = x\n    for i in range(0):\n        a = a + 1\n    return i + a"
    )
    assert vec(np.asarray([1.0]))[0] == 2.0


# 17: constant scalar math works on strict backends


def test_constant_scalar_math_on_strict_backend() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return x + math.sqrt(4.0)"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 3.0
