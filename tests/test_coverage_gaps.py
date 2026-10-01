"""Coverage-closing tests for error paths and fold branches."""

from __future__ import annotations

import importlib.util
import math
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from vectorizer import vectorize
from vectorizer._errors import VectorizationError
from vectorizer._extract import extract_function
from vectorizer._ir import Binding, BinOp, Call, Literal, Program, Ref, SSAEnv, UnaryOp
from vectorizer._optimize import const_fold, cse, optimize
from vectorizer._optimize import dce as dce_pass

_tmp = tempfile.TemporaryDirectory(prefix="vec_cov_")
_TMPDIR = Path(_tmp.name)
_seq = __import__("itertools").count()


def make_fn(body: str) -> Callable[..., Any]:
    path = _TMPDIR / f"snippet_{next(_seq)}.py"
    path.write_text("import math\n\n\ndef subject(x, y=2.0):\n" + body + "\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.subject


def lit(v: object) -> Literal:
    if isinstance(v, bool):
        return Literal(v, "bool")
    if isinstance(v, int):
        return Literal(v, "int")
    return Literal(float(v), "float")  # type: ignore[arg-type]


# ------------------------------------------------------- const-fold branches


@pytest.mark.parametrize(
    ("op", "a", "b", "expected"),
    [
        ("floordiv", 7, 2, 3),
        ("mod", 7, 3, 1),
        ("and", 6, 3, 2),
        ("or", 6, 3, 7),
        ("xor", 6, 3, 5),
        ("lshift", 1, 3, 8),
        ("rshift", 8, 2, 2),
        ("sub", 2, 5, -3),
        ("mul", 2.5, 4, 10.0),
    ],
)
def test_fold_binop_ops(op: str, a: float, b: float, expected: float) -> None:
    out = const_fold(Program((), (), BinOp(op, lit(a), lit(b))))
    assert out.result == lit(expected)


def test_fold_unary_ops_all() -> None:
    assert const_fold(Program((), (), UnaryOp("invert", lit(0)))).result == lit(-1)
    assert const_fold(Program((), (), UnaryOp("pos", lit(-2)))).result == lit(-2)
    assert const_fold(Program((), (), UnaryOp("neg", lit(2.5)))).result == lit(-2.5)
    assert const_fold(Program((), (), UnaryOp("not", lit(True)))).result == lit(False)


def test_fold_skips_bad_ops() -> None:
    # float bitwise -> TypeError at runtime -> no fold
    expr = BinOp("and", lit(1.5), lit(2.5))
    assert const_fold(Program((), (), expr)).result == expr
    # unary invert on float -> no fold
    expr = UnaryOp("invert", lit(1.5))
    assert const_fold(Program((), (), expr)).result == expr
    # unknown unary op passes through
    expr = UnaryOp("weird", lit(1))
    assert const_fold(Program((), (), expr)).result == expr
    # floordiv by zero
    expr = BinOp("floordiv", lit(1), lit(0))
    assert const_fold(Program((), (), expr)).result == expr


def test_fold_zero_times_infinity() -> None:
    # inf * 0 -> nan: exact IEEE fold
    out = const_fold(Program((), (), BinOp("mul", lit(math.inf), lit(0.0))))
    assert math.isnan(out.result.value)  # type: ignore[attr-defined]


def test_dce_keeps_result_only_binding() -> None:
    p = Program(("x",), (Binding("a", Call("abs", (Ref("x"),))),), Ref("a"))
    assert dce_pass(p).bindings == p.bindings


def test_cse_stops_at_fixpoint() -> None:
    inner = Call("sqrt", (Ref("x"),))
    p = Program(("x",), (), BinOp("add", inner, inner))
    out = cse(p, SSAEnv(set()))
    assert out.bindings == (Binding("t_1", inner),)
    assert out.result == BinOp("add", Ref("t_1"), Ref("t_1"))


def test_optimize_no_user_names() -> None:
    p = Program(("x",), (Binding("a", Ref("x")),), Ref("a"))
    assert optimize(p).result == Ref("a")


# ------------------------------------------------------------ extract paths


def test_async_function_rejected() -> None:
    path = _TMPDIR / "async_snippet.py"
    path.write_text("async def subject(x):\n    return x\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(VectorizationError, match="async"):
        extract_function(mod.subject)


def test_kwonly_without_default() -> None:
    path = _TMPDIR / "kwonly_snippet.py"
    path.write_text("def subject(x, *, scale):\n    return x * scale\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    info = extract_function(mod.subject)
    assert info.params[1].kind == "kwonly"
    assert not info.params[1].has_default


def test_expr_lambda_top_level() -> None:
    path = _TMPDIR / "expr_lambda.py"
    path.write_text("(lambda x: x + 1)\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # nothing to grab; just ensure parse path works via direct lambda
    info = extract_function(lambda x: x + 1.0)  # type: ignore[arg-type]
    assert info.params[0].name == "x"


def test_no_target_in_source_rejected() -> None:
    path = _TMPDIR / "no_target.py"
    path.write_text("X = 1\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(VectorizationError):
        extract_function(mod.X)  # type: ignore[arg-type]


# --------------------------------------------------------- validator paths


@pytest.mark.parametrize(
    "body",
    [
        "    y = 1\n    return y",
    ],
)
def test_validator_ok(body: str) -> None:
    extract_function(make_fn(body))


# ------------------------------------------------------------ lower errors


def test_calling_math_module_rejected() -> None:
    with pytest.raises(VectorizationError, match="math module"):
        vectorize(make_fn("    return math(x)"))


def test_math_const_called_rejected() -> None:
    with pytest.raises(VectorizationError, match="cannot be called"):
        vectorize(make_fn("    return math.pi(x)"))


def test_calling_shadowed_builtin_rejected() -> None:
    with pytest.raises(VectorizationError, match="calling variable"):
        vectorize(make_fn("    abs = x\n    return abs(x)"))


def test_statements_after_return_in_branch_rejected() -> None:
    with pytest.raises(VectorizationError, match="after return"):
        vectorize(make_fn("    if x > 0:\n        return x\n        y = 1\n    return 0.0"))


def test_deeply_nested_boolop_chain() -> None:
    vec = vectorize(make_fn("    return x and y and x and y"))
    import numpy as np

    xs = np.asarray([0.0, 1.0, 2.0])
    ys = np.asarray([3.0, 0.0, 5.0])
    out = vec(xs, ys)
    assert np.allclose(out, [0.0, 0.0, 5.0])


def test_not_all_paths_return_via_branch_fallthrough() -> None:
    with pytest.raises(VectorizationError, match="paths return"):
        vectorize(
            make_fn("    if x > 0:\n        return 1.0\n    if x < -1:\n        return 2.0\n")
        )


def test_maybe_unbound_after_nested_branch() -> None:
    with pytest.raises(VectorizationError, match="may be unbound"):
        vectorize(
            make_fn(
                "    if x > 0:\n"
                "        if x > 1:\n"
                "            z = 1.0\n"
                "        else:\n"
                "            z = 2.0\n"
                "    else:\n"
                "        y = 3.0\n"
                "    return z"
            )
        )
