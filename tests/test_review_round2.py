"""Regression tests for the adversarial-review findings (round 2)."""

from __future__ import annotations

import importlib.util
import math
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from support import TMPDIR, make_fn

from vectorizer import vectorize


def vec_of(body: str, extra: str = "") -> tuple[Callable[..., Any], Callable[..., Any]]:
    fn = make_fn(body, extra)
    return vectorize(fn), fn


# 1: loop-carried kinds mix across iterations


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


# 2: nested loop targets are loop-carried dependencies


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


# 3 + 7 + 8: protect_domains never touches live lanes, never warns


def test_protect_domains_preserves_subnormal_live_lanes() -> None:
    vec = vectorize(make_fn("    return math.log(x) if x > 0 else 0.0"), protect_domains=True)
    got = vec(np.asarray([1e-310]))
    assert got[0] == math.log(1e-310)


def test_protect_domains_open_boundaries_never_warn() -> None:
    vec = vectorize(make_fn("    return math.log1p(x) if x > -1 else 0.0"), protect_domains=True)
    with np.errstate(all="raise"):
        got = vec(np.asarray([-1.0, 0.0]))
    assert np.allclose(got, [0.0, 0.0])


def test_protect_domains_atanh_dead_lanes() -> None:
    vec = vectorize(
        make_fn("    return math.atanh(x) if -1 < x < 1 else 0.0"), protect_domains=True
    )
    with np.errstate(all="raise"):
        got = vec(np.asarray([2.0, 0.5, -2.0]))
    assert np.allclose(got, [0.0, math.atanh(0.5), 0.0])


def test_protect_domains_follows_branch_bindings() -> None:
    vec = vectorize(
        make_fn(
            "    if x > 0:\n        y = math.sqrt(x)\n    else:\n        y = 0.0\n    return y"
        ),
        protect_domains=True,
    )
    with np.errstate(all="raise"):
        got = vec(np.asarray([-1.0, 1.0]))
    assert np.allclose(got, [0.0, 1.0])


def test_protect_domains_propagates_to_helpers() -> None:
    # the helper's own internal where gets protected; note that a helper
    # called only inside a caller's where-branch cannot know its lanes are
    # dead (cross-function liveness is a documented limitation)
    fn = make_fn(
        "    return helper(x) * 2.0",
        extra="def helper(v):\n    return math.sqrt(v) if v > 0 else 0.0\n",
    )
    vec = vectorize(fn, protect_domains=True)
    with np.errstate(all="raise"):
        got = vec(np.asarray([-1.0, 4.0]))
    assert np.allclose(got, [0.0, 4.0])


def test_protect_domains_multiple_uses_or_liveness() -> None:
    # y_1 is used in two wheres with different (overlapping) conditions:
    # only lanes dead for BOTH uses are clampable; live lanes keep exact
    # values. Live region = (x > 0) or (x >= 0) = x >= 0; dead = x < 0.
    vec = vectorize(
        make_fn(
            "    y = math.sqrt(x)\n"
            "    a = y if x > 0 else 0.0\n"
            "    b = y if x >= 0 else 1.0\n"
            "    return a + b"
        ),
        protect_domains=True,
    )
    with np.errstate(all="raise"):
        got = vec(np.asarray([-4.0, 4.0]))
    assert np.allclose(got, [1.0, 4.0])


# 4: lambda identification


def test_lambdas_differing_only_in_names() -> None:
    path = TMPDIR / "lam_names.py"
    path.write_text("import math\nf1, f2 = (lambda x: math.sin(x)), (lambda x: math.cos(x))\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert np.isclose(vectorize(mod.f2)(np.asarray([1.0]))[0], math.cos(1.0))
    assert np.isclose(vectorize(mod.f1)(np.asarray([1.0]))[0], math.sin(1.0))


def test_lambdas_differing_only_in_defaults() -> None:
    path = TMPDIR / "lam_defaults.py"
    path.write_text("g1, g2 = (lambda x, y=1: x + y), (lambda x, y=2: x + y)\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert vectorize(mod.g2)(np.asarray([1.0]))[0] == 3.0
    assert vectorize(mod.g1)(np.asarray([1.0]))[0] == 2.0


# 6 + 11: namespace allocation and caller-name reservations


def test_namespace_var_rebinding_no_collision() -> None:
    path = TMPDIR / "xp_rebind.py"
    path.write_text("import math\n\n\ndef subject(xp):\n    xp = xp + 1\n    return math.sin(xp)\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vec = vectorize(mod.subject)
    assert np.allclose(vec(xp=np.asarray([1.0])), [math.sin(2.0)])


def test_reserved_caller_name_param_not_renamed() -> None:
    path = TMPDIR / "reserved_caller.py"
    path.write_text(
        "def reserved_caller(reserved_caller_vec):\n    return reserved_caller_vec + 1\n"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vec = vectorize(mod.reserved_caller)
    assert np.allclose(vec(reserved_caller_vec=np.asarray([1.0])), [2.0])


# 9: literal bindings and strict-backend scalar arguments


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


# 10: strict-backend scalars in verify= and fallback


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
