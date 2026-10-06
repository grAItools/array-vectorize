"""Helper vectorization and caching, name shadowing, rejections."""

from __future__ import annotations

import numpy as np
import pytest
from support import make_fn, make_module

from array_vectorize import VectorizationError, vectorize

# ---------------------------------------------------------------- helper calls


def test_helper_call_composition() -> None:
    mod = make_module(
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
    vec = vectorize(mod.norm)
    xs = np.asarray([3.0, 4.0])
    got = vec(xs)
    assert np.allclose(got, np.sqrt(2 * xs**2))
    assert "square_vec(" in vec.source
    # the helper itself is vectorized and memoized
    from array_vectorize.api import _HELPER_CACHE

    assert (mod.square, False, None) in _HELPER_CACHE


def test_helper_partial_application() -> None:
    mod = make_module(
        "def scale(y, k):\n"
        "    return y * k\n"
        "\n"
        "\n"
        "def twice(x):\n"
        "    return scale(x, 2.0) + scale(x, 2.0)\n"
    )
    vec = vectorize(mod.twice)
    assert np.allclose(vec(np.asarray([1.0, 2.0])), [4.0, 8.0])


def test_helper_with_loop_calling_helper() -> None:
    mod = make_module(
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
    vec = vectorize(mod.combined)
    xs = np.asarray([2.0])
    ys = np.asarray([3.0])
    assert np.allclose(vec(xs, ys), [2 * 3 * 3 * 2])
    # the loop lives in the helper's generated source, not the caller's
    from array_vectorize.api import _HELPER_CACHE

    helper_src = _HELPER_CACHE[(mod.addmul, False, None)].source
    assert "for i in range(0, 3):" in helper_src
    assert "addmul_vec" in vec.source


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


def test_helper_cache_protect_flag() -> None:
    mod = make_module(
        "import math\n\n\n"
        "def inner(x):\n"
        "    return math.sqrt(x) if x > 0 else 0.0\n\n\n"
        "def outer(x):\n"
        "    return inner(x)\n"
    )
    vectorize(mod.outer)  # caches the unprotected inner
    protected = vectorize(mod.outer, protect_domains=True)
    with np.errstate(all="raise"):
        got = protected(np.asarray([-1.0, 4.0]))
    assert np.allclose(got, [0.0, 2.0])


def test_recursive_helper_rejected() -> None:
    mod = make_module(
        "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n"
    )
    with pytest.raises(VectorizationError, match="recursive"):
        vectorize(mod.fib)


def test_helper_keyword_call_rejected() -> None:
    fn = make_fn("    return helper(x, y=9)", extra="def helper(x, y=2):\n    return x + y\n")
    with pytest.raises(VectorizationError, match="keyword"):
        vectorize(fn)
