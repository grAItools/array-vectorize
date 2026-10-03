"""M3 composition tests: loops (D6), helper calls (D7), fallback mode (T10)."""

from __future__ import annotations

import importlib.util
import itertools
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vectorizer import VectorizationError, vectorize

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


# ------------------------------------------------------------------- loops


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


# ------------------------------------------------------------------ helpers


def test_helper_call_composition() -> None:
    path = _TMPDIR / "helpers_mod.py"
    path.write_text(
        "import math\n"
        "\n"
        "\n"
        "def square(y):\n"
        "    return y * y\n"
        "\n"
        "\n"
        "def norm(x):\n"
        "    return math.sqrt(square(x) + square(x))\n"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vec = vectorize(mod.norm)
    xs = np.asarray([3.0, 4.0])
    got = vec(xs)
    assert np.allclose(got, np.sqrt(2 * xs**2))
    assert "square_vec(" in vec.source
    # the helper itself is vectorized and memoized
    from vectorizer import _HELPER_CACHE

    assert (mod.square, False) in _HELPER_CACHE


def test_helper_partial_application() -> None:
    path = _TMPDIR / "helpers_partial.py"
    path.write_text(
        "def scale(y, k):\n"
        "    return y * k\n"
        "\n"
        "\n"
        "def twice(x):\n"
        "    return scale(x, 2.0) + scale(x, 2.0)\n"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vec = vectorize(mod.twice)
    assert np.allclose(vec(np.asarray([1.0, 2.0])), [4.0, 8.0])


def test_recursive_helper_rejected() -> None:
    path = _TMPDIR / "recursive_mod.py"
    path.write_text(
        "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(VectorizationError, match="recursive"):
        vectorize(mod.fib)


def test_helper_with_loop_calling_helper() -> None:
    path = _TMPDIR / "loop_helper.py"
    path.write_text(
        "def addmul(a, b):\n"
        "    s = 0.0\n"
        "    for i in range(3):\n"
        "        s = s + a * b\n"
        "    return s\n"
        "\n"
        "\n"
        "def combined(x, y):\n"
        "    return addmul(x, y) + addmul(y, x)\n"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vec = vectorize(mod.combined)
    xs = np.asarray([2.0])
    ys = np.asarray([3.0])
    assert np.allclose(vec(xs, ys), [2 * 3 * 3 * 2])
    # the loop lives in the helper's generated source, not the caller's
    from vectorizer import _HELPER_CACHE

    helper_src = _HELPER_CACHE[(mod.addmul, False)].source
    assert "for i in range(0, 3):" in helper_src
    assert "addmul_vec" in vec.source


# ----------------------------------------------------------------- fallback


def test_fallback_used_on_unsupported() -> None:
    def bad(x: float) -> float:
        while x > 0:
            x = x - 1
        return x

    with pytest.warns(UserWarning, match="falling back"):
        vec = vectorize(bad, fallback=True)
    xs = np.asarray([5.0, -2.0])
    assert np.allclose(vec(xs), [0.0, -2.0])


def test_strict_false_implies_fallback() -> None:
    def bad(x: float) -> float:
        while x > 0:
            x = x - 1
        return x

    with pytest.warns(UserWarning, match="falling back"):
        vec = vectorize(bad, strict=False)
    assert np.allclose(vec(np.asarray([5.0, -1.0])), [0.0, -1.0])


def test_fallback_no_vectorization_error_raised() -> None:
    def bad(x: float) -> float:
        return x

    bad.__code__ = bad.__code__  # keep linters quiet

    # sanity: same function without fallback still raises
    def also_bad(x: float) -> float:
        while x > 0:
            x = x - 1
        return x

    with pytest.raises(VectorizationError):
        vectorize(also_bad)


def test_fallback_broadcasts_args() -> None:
    def bad(x: float, y: float) -> float:
        while x > 0:
            x = x - y
        return x

    with pytest.warns(UserWarning):
        vec = vectorize(bad, fallback=True)
    xs = np.asarray([[3.0]])  # broadcasts against ys
    ys = np.asarray([[1.0, 2.0]])
    assert np.allclose(vec(xs, ys), [[0.0, -1.0]])


def test_fallback_signature_and_source() -> None:
    def bad(x: float, scale: float = 2.0) -> float:
        while x > 0:
            x = x - scale
        return x

    import inspect

    with pytest.warns(UserWarning):
        vec = vectorize(bad, fallback=True)
    assert list(inspect.signature(vec).parameters) == ["x", "scale"]
    assert "fallback" in vec.source
