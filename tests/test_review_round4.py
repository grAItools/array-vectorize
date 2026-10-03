"""Regression tests for the adversarial-review findings (round 4)."""

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

_tmp = tempfile.TemporaryDirectory(prefix="vec_rev4_")
_TMPDIR = Path(_tmp.name)
_seq = itertools.count()


def make_fn(body: str, defaults: str = "x, y=2.0") -> Callable[..., Any]:
    path = _TMPDIR / f"snippet_{next(_seq)}.py"
    path.write_text("import math\n\n\ndef subject(" + defaults + "):\n" + body + "\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.subject


# 1: bool arithmetic keeps integer semantics


def test_bool_plus_large_int_exact() -> None:
    vec = vectorize(make_fn("    return (x > 0) + 9007199254740992"))
    got = vec(np.asarray([1.0]))
    assert got[0] == 9007199254740993


def test_bool_arith_then_bitwise() -> None:
    vec = vectorize(make_fn("    return ((x > 0) + (x > 0)) & 1"))
    got = vec(np.asarray([1.0]))
    assert got[0] == 0


# 2: chained loop-kind propagation reaches a fixed point


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


# 3: min/max literals dtype-match their Ref sibling at runtime


def test_min_literal_exact_large_int() -> None:
    vec = vectorize(make_fn("    return min(x, 9007199254740993)"))
    got = vec(np.asarray([9007199254740994], dtype=np.int64))
    assert got[0] == 9007199254740993


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


# 4: mixed scalar-default arithmetic keeps scalar promotion


def test_int_default_scalar_promotion_strict() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    return x + y", defaults="x, y=1"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 2.0


# 5: computed local scalars and loop-index expressions


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


# 6: protect_domains forward references to loop-body bindings


def test_protect_domains_loop_body_binding_guard() -> None:
    body = (
        "    y = math.sqrt(x)\n"
        "    for i in range(1):\n"
        "        c = x > 0\n"
        "    return y if c else 0.0"
    )
    plain = vectorize(make_fn(body))(np.asarray([4.0]))
    protected = vectorize(make_fn(body), protect_domains=True)(np.asarray([4.0]))
    assert plain[0] == 2.0
    assert protected[0] == 2.0


# 7: all-literal min/max folds through literal-backed references


def test_all_literal_minmax_refs_fold() -> None:
    import array_api_strict as xps

    vec = vectorize(make_fn("    a = 1.0\n    b = 2.0\n    return x + min(a, b)"))
    got = vec(xps.asarray([1.0]))
    assert float(got[0]) == 2.0


# 8: verify handles constant-return functions


def test_verify_constant_return() -> None:
    vec = vectorize(make_fn("    return 3.0"), verify=(np.asarray([1.0, 2.0]),))
    got = vec(np.asarray([1.0, 2.0]))
    assert np.allclose(np.broadcast_to(np.asarray(got), (2,)), [3.0, 3.0])
