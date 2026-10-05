"""Closures, helper vectorization and caching, name allocation, lambda identification, CSE temps."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from support import TMPDIR, just_vec, make_fn, vec_of

from array_vectorize import VectorizationError, vectorize

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


_TMPDIR = TMPDIR


# ---- from test_behavior (git history: tests/test_behavior.py)


def test_closure_scalar() -> None:
    assert np.allclose(vfn("closure_scalar")(X), X * CORPUS.SCALE)


def test_closure_array() -> None:
    got = vfn("closure_array")(np.asarray([1.0, 1.0, 1.0]))
    assert np.allclose(got, np.asarray([2.0, 3.0, 4.0]))


def test_kwonly_defaults() -> None:
    expected = X * 2.0 + 0.5
    assert np.allclose(vfn("kwonly_defaults")(X), expected)
    expected = X * 4.0 + 1.0
    assert np.allclose(vfn("kwonly_defaults")(X, scale=4.0, bias=1.0), expected)


def test_reserved_param() -> None:
    assert np.allclose(vfn("reserved_param")(X), X + 1)


def test_cse_opportunity() -> None:
    s = np.sqrt(np.abs(X))
    expected = s * s + s
    assert np.allclose(vfn("cse_opportunity")(np.abs(X)), expected)


# ---- from test_composition (git history: tests/test_composition.py)


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
    from array_vectorize.api import _HELPER_CACHE

    assert (mod.square, False, None) in _HELPER_CACHE


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
    from array_vectorize.api import _HELPER_CACHE

    helper_src = _HELPER_CACHE[(mod.addmul, False, None)].source
    assert "for i in range(0, 3):" in helper_src
    assert "addmul_vec" in vec.source


# ---- from test_review_round1 (git history: tests/test_review_round1.py)


def test_cse_temp_does_not_collide_with_ssa_bindings() -> None:
    vec, fn = vec_of("    t = x + 1\n    t = t + 1\n    return t + math.sin(x) + math.sin(x)")
    xs = np.asarray([1.0])
    assert np.isclose(vec(xs)[0], fn(1.0))


def test_signed_zero_not_cse_merged() -> None:
    vec = just_vec("    return math.copysign(x, 0.0) + math.copysign(x, -0.0)")
    assert np.allclose(vec(np.asarray([1.0])), [0.0])


def test_helper_keyword_call_rejected() -> None:
    fn = make_fn("    return helper(x, y=9)", extra="def helper(x, y=2):\n    return x + y\n")
    with pytest.raises(VectorizationError, match="keyword"):
        vectorize(fn)


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


def test_multiple_lambdas_on_one_line() -> None:
    path = TMPDIR / "multi_lambda.py"
    path.write_text("f1, f2 = (lambda x: x + 1), (lambda x: x + 2)\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert vectorize(mod.f2)(np.asarray([1.0]))[0] == 3.0
    assert vectorize(mod.f1)(np.asarray([1.0]))[0] == 2.0


def test_hidden_params_preserve_kwonly_defaults() -> None:
    path = TMPDIR / "kwonly_closure.py"
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


def test_reserved_xp_param_keyword_call() -> None:
    path = TMPDIR / "xp_param.py"
    path.write_text("import numpy as np\n\n\ndef subject(xp):\n    return xp + 1\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vec = vectorize(mod.subject)
    assert np.allclose(vec(xp=np.asarray([1.0])), [2.0])
    assert "def subject_vec(xp):" in vec.source


# ---- from test_review_round2 (git history: tests/test_review_round2.py)


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


# ---- from test_review_round3 (git history: tests/test_review_round3.py)


def test_lambda_signed_zero_constants() -> None:
    path = TMPDIR / "lam_zeros.py"
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
    path = TMPDIR / "lam_types.py"
    path.write_text("g1, g2, g3 = (lambda x: x + 1), (lambda x: x + True), (lambda x: x + 1.0)\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert vectorize(mod.g2)(np.asarray([1.0]))[0] == 2.0
    assert vectorize(mod.g3)(np.asarray([1.0]))[0] == 2.0


# ---- from test_review_round6 (git history: tests/test_review_round6.py)


def test_helper_name_shadowed_by_param() -> None:
    vec = vectorize(make_fn("    return min(_vec_common_dtype, 3)", defaults="_vec_common_dtype"))
    got = vec(np.asarray([1.0]))
    assert got[0] == 1.0


def test_helper_names_shadowed_by_params() -> None:
    vec = vectorize(
        make_fn(
            "    return min(_vec_arith_dtype, _vec_common_dtype) + (_vec_arith_dtype > 0)",
            defaults="_vec_arith_dtype, _vec_common_dtype=2.0",
        )
    )
    got = vec(np.asarray([1.0]))
    assert got[0] == 2.0  # min(1, 2) + True


def test_helper_cache_protect_flag() -> None:
    path = TMPDIR / "cache_protect.py"
    path.write_text(
        "import math\n\n\n"
        "def inner(x):\n"
        "    return math.sqrt(x) if x > 0 else 0.0\n\n\n"
        "def outer(x):\n"
        "    return inner(x)\n"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vectorize(mod.outer)  # caches the unprotected inner
    protected = vectorize(mod.outer, protect_domains=True)
    with np.errstate(all="raise"):
        got = protected(np.asarray([-1.0, 4.0]))
    assert np.allclose(got, [0.0, 2.0])
