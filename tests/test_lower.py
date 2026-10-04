"""Lowering tests: AST -> IR shapes (plan §5/§6), straight-line (M1) + control flow (M2)."""

from __future__ import annotations

import importlib.util
import itertools
import math
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vectorizer._errors import VectorizationError
from vectorizer._extract import extract_function
from vectorizer._ir import (
    Binding,
    BinOp,
    Call,
    Compare,
    DType,
    FuncCall,
    Literal,
    Logical,
    Program,
    Ref,
    UnaryOp,
    Where,
)
from vectorizer._lower import lower_function

_tmp = tempfile.TemporaryDirectory(prefix="vec_lower_")
_TMPDIR = Path(_tmp.name)
_seq = itertools.count()

GLOBAL_K = 3
GLOBAL_ARR = np.asarray([10.0, 20.0])


def make_fn(body: str, extra_globals: str = "") -> Callable[..., Any]:
    path = _TMPDIR / f"snippet_{next(_seq)}.py"
    path.write_text(
        "import math\n"
        "from math import exp\n"
        "GLOBAL_K = 3\n"
        "GLOBAL_ARR = __import__('numpy').asarray([10.0, 20.0])\n"
        + extra_globals
        + "\n\n\ndef subject(x, y=2.0):\n"
        + body
        + "\n"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.subject


def lower(body: str) -> Program:
    return lower_function(extract_function(make_fn(body))).program


# ------------------------------------------------------------- M1: expressions


def test_const_and_binop() -> None:
    assert lower("    return x + 1") == Program(
        ("x", "y"), (), BinOp("add", Ref("x"), Literal(1, "int"))
    )


def test_arith_ops_map() -> None:
    p = lower("    return x - y * 2 / 3")
    assert p.result == BinOp(
        "sub", Ref("x"), BinOp("div", BinOp("mul", Ref("y"), Literal(2, "int")), Literal(3, "int"))
    )


def test_pow_floordiv_mod_bitwise() -> None:
    assert lower("    return x ** 2").result == BinOp("pow", Ref("x"), Literal(2, "int"))
    assert lower("    return x // 2").result == BinOp("floordiv", Ref("x"), Literal(2, "int"))
    assert lower("    return x % 2").result == BinOp("mod", Ref("x"), Literal(2, "int"))
    assert lower("    return x & 1").result == BinOp("and", Ref("x"), Literal(1, "int"))
    assert lower("    return x | 1").result == BinOp("or", Ref("x"), Literal(1, "int"))
    assert lower("    return x ^ 1").result == BinOp("xor", Ref("x"), Literal(1, "int"))
    assert lower("    return x << 1").result == BinOp("lshift", Ref("x"), Literal(1, "int"))
    assert lower("    return x >> 1").result == BinOp("rshift", Ref("x"), Literal(1, "int"))


def test_unary_ops() -> None:
    assert lower("    return -x").result == UnaryOp("neg", Ref("x"))
    assert lower("    return +x").result == UnaryOp("pos", Ref("x"))
    assert lower("    return ~x").result == UnaryOp("invert", Ref("x"))
    assert lower("    return not (x > 0)").result == UnaryOp(
        "not", Compare("gt", Ref("x"), Literal(0, "int"))
    )


def test_bool_literals_typed() -> None:
    # bool literal in arithmetic: both operands are cast to the runtime
    # common dtype (bools need a numeric representation; backends
    # saturate or reject bool arithmetic)
    dt = FuncCall(
        "_vec_arith_dtype", (Ref("xp"), Literal(1, "int"), Ref("x"), Literal(True, "bool"))
    )
    assert lower("    return x + True").result == BinOp(
        "add",
        Call("astype", (Call("asarray", (Ref("x"),)), dt)),
        Call("astype", (Call("asarray", (Literal(True, "bool"),)), dt)),
    )
    assert lower("    return 1.5").result == Literal(1.5, "float")


def test_math_attribute_call() -> None:
    assert lower("    return math.sqrt(x)").result == Call(
        "sqrt", (Call("astype", (Call("asarray", (Ref("x"),)), DType("float64"))),)
    )
    assert lower("    return math.atan2(x, y)").result == Call(
        "atan2",
        (
            Call("astype", (Call("asarray", (Ref("x"),)), DType("float64"))),
            Call("astype", (Call("asarray", (Ref("y"),)), DType("float64"))),
        ),
    )
    assert lower("    return math.trunc(x)").result == Call(
        "astype", (Call("asarray", (Ref("x"),)), DType("int64"))
    )
    assert lower("    return math.pow(x, 2)").result == Call(
        "pow",
        (Call("astype", (Call("asarray", (Ref("x"),)), DType("float64"))), Literal(2.0, "float")),
    )


def test_from_math_import_call() -> None:
    assert lower("    return exp(x)").result == Call(
        "exp", (Call("astype", (Call("asarray", (Ref("x"),)), DType("float64"))),)
    )


def test_math_constants() -> None:
    assert lower("    return math.pi").result == Literal(math.pi, "float")
    assert lower("    return math.inf").result == Literal(math.inf, "float")


def test_builtins() -> None:
    assert lower("    return abs(x)").result == Call("abs", (Call("asarray", (Ref("x"),)),))
    assert lower("    return round(x)").result == Call("round", (Call("asarray", (Ref("x"),)),))
    assert lower("    return min(x, y)").result == FuncCall(
        "_vec_minmax",
        (
            Ref("xp"),
            Literal(True, "bool"),
            Call("asarray", (Ref("x"),)),
            Call("asarray", (Ref("y"),)),
        ),
    )
    assert lower("    return max(x, y, 2.0)").result == FuncCall(
        "_vec_minmax",
        (
            Ref("xp"),
            Literal(False, "bool"),
            Call("asarray", (Ref("x"),)),
            Call("asarray", (Ref("y"),)),
            Literal(2.0, "float"),
        ),
    )
    assert lower("    return int(x)").result == Call(
        "astype", (Call("asarray", (Ref("x"),)), DType("int64"))
    )
    assert lower("    return float(x)").result == Call(
        "astype", (Call("asarray", (Ref("x"),)), DType("float64"))
    )
    assert lower("    return bool(x)").result == Call(
        "astype", (Call("asarray", (Ref("x"),)), DType("bool"))
    )


def test_comparisons_and_chains() -> None:
    assert lower("    return x < y").result == Compare("lt", Ref("x"), Ref("y"))
    assert lower("    return 0 < x < 1").result == Logical(
        "and",
        (
            Compare("lt", Literal(0, "int"), Ref("x")),
            Compare("lt", Ref("x"), Literal(1, "int")),
        ),
    )


def test_boolop_bool_operands_use_logical() -> None:
    assert lower("    return (x < 0) and (y > 0)").result == Logical(
        "and",
        (Compare("lt", Ref("x"), Literal(0, "int")), Compare("gt", Ref("y"), Literal(0, "int"))),
    )


def test_boolop_numeric_value_select() -> None:
    assert lower("    return x and y").result == Where(
        Compare("ne", Ref("x"), Literal(0, "int")), Ref("y"), Ref("x")
    )
    assert lower("    return x or y").result == Where(
        Compare("ne", Ref("x"), Literal(0, "int")), Ref("x"), Ref("y")
    )


def test_boolop_numeric_chain_uses_temp() -> None:
    p = lower("    return x and y and x")
    assert len(p.bindings) == 1
    temp = p.bindings[0]
    assert temp.name == "v_1"
    assert temp.expr == Where(Compare("ne", Ref("x"), Literal(0, "int")), Ref("y"), Ref("x"))
    assert p.result == Where(Compare("ne", Ref("v_1"), Literal(0, "int")), Ref("x"), Ref("v_1"))


def test_ternary_coerces_numeric_condition() -> None:
    assert lower("    return x if x else y").result == Where(
        Compare("ne", Ref("x"), Literal(0, "int")), Ref("x"), Ref("y")
    )
    assert lower("    return x if x > 0 else y").result == Where(
        Compare("gt", Ref("x"), Literal(0, "int")), Ref("x"), Ref("y")
    )


# ------------------------------------------------------------ M1: statements


def test_assign_and_rebind() -> None:
    p = lower("    z = x * 2\n    z = z + 1\n    return z")
    assert p.bindings == (
        Binding("z", BinOp("mul", Ref("x"), Literal(2, "int"))),
        Binding("z_1", BinOp("add", Ref("z"), Literal(1, "int"))),
    )
    assert p.result == Ref("z_1")


def test_augassign() -> None:
    p = lower("    z = x\n    z += 1\n    z *= 2\n    return z")
    assert p.bindings == (
        Binding("z", Ref("x")),
        Binding("z_1", BinOp("add", Ref("z"), Literal(1, "int"))),
        Binding("z_2", BinOp("mul", Ref("z_1"), Literal(2, "int"))),
    )
    assert p.result == Ref("z_2")


def test_rebind_skips_user_names() -> None:
    p = lower("    x_1 = 0\n    x = x + 1\n    x = x + 1\n    return x + x_1")
    names = [b.name for b in p.bindings]
    assert names[1:] == ["x_2", "x_3"]  # x_1 taken by the user variable


def test_closure_scalar_frozen() -> None:
    assert lower("    return x * GLOBAL_K").result == BinOp("mul", Ref("x"), Literal(3, "int"))


def test_closure_array_hidden_param() -> None:
    lf = lower_function(extract_function(make_fn("    return x + GLOBAL_ARR")))
    assert lf.program.params == ("x", "y", "GLOBAL_ARR")
    assert lf.program.result == BinOp("add", Ref("x"), Ref("GLOBAL_ARR"))
    assert len(lf.hidden_params) == 1
    name, default = lf.hidden_params[0]
    assert name == "GLOBAL_ARR"
    assert np.array_equal(default, [10.0, 20.0])


def test_reserved_param_name_kept_namespace_renamed() -> None:
    # a parameter named 'xp' keeps its name (keyword calls work); the
    # generated namespace variable renames itself instead
    path = _TMPDIR / f"snippet_{next(_seq)}.py"
    path.write_text("def subject(xp):\n    return xp + 1\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lf = lower_function(extract_function(mod.subject))
    assert lf.param_names == ["xp"]
    assert lf.namespace_var == "xp_1"
    assert lf.program.result == BinOp("add", Ref("xp"), Literal(1, "int"))


# ------------------------------------------------------------- M1: rejections


@pytest.mark.parametrize(
    "body",
    [
        "    return len(x)",  # unsupported builtin
        "    return round(x, 2)",  # decimals not standard
        "    return min(x)",  # arity
        "    return math.hypot(x)",  # arity
        "    return unknown_fn(x)",  # unknown function
    ],
)
def test_expression_rejections(body: str) -> None:
    with pytest.raises(VectorizationError):
        lower(body)


def test_reference_before_assignment() -> None:
    with pytest.raises(VectorizationError, match="before assignment"):
        lower("    z = z + 1\n    return z")


def test_closure_scalar_read_after_local_assign_rejected() -> None:
    # GLOBAL_K is assigned locally -> reads before the assignment are unbound
    with pytest.raises(VectorizationError, match="before assignment"):
        lower("    z = GLOBAL_K + 1\n    GLOBAL_K = 2\n    return z")


def test_calling_variable_rejected() -> None:
    with pytest.raises(VectorizationError, match="calling variable"):
        lower("    f = x\n    return f(x)")


# -------------------------------------------------------------- M2: if/else


def test_if_else_merge() -> None:
    p = lower("    if x > 0:\n        r = x\n    else:\n        r = 0.0\n    return r")
    assert [b.name for b in p.bindings] == ["r_1", "r_2", "r"]
    assert p.bindings[0] == Binding("r_1", Ref("x"))
    assert p.bindings[1] == Binding("r_2", Literal(0.0, "float"))
    assert p.bindings[2] == Binding(
        "r", Where(Compare("gt", Ref("x"), Literal(0, "int")), Ref("r_1"), Ref("r_2"))
    )
    assert p.result == Ref("r")


def test_if_else_merge_with_prior_binding() -> None:
    p = lower("    r = 0.0\n    if x > 0:\n        r = x\n    else:\n        r = 1.0\n    return r")
    assert [b.name for b in p.bindings] == ["r", "r_1", "r_2", "r_3"]
    assert p.result == Ref("r_3")


def test_if_without_else_and_prior_binding() -> None:
    p = lower("    r = 0.0\n    if x > 0:\n        r = x\n    return r")
    # then binds r_1; else keeps r; merge where(cond, r_1, r)
    assert [b.name for b in p.bindings] == ["r", "r_1", "r_2"]
    assert p.bindings[2].expr == Where(
        Compare("gt", Ref("x"), Literal(0, "int")), Ref("r_1"), Ref("r")
    )
    assert p.result == Ref("r_2")


def test_elif_chain() -> None:
    p = lower(
        "    if x > 1:\n"
        "        r = 2.0\n"
        "    elif x > 0:\n"
        "        r = 1.0\n"
        "    else:\n"
        "        r = 0.0\n"
        "    return r"
    )
    names = [b.name for b in p.bindings]
    assert names == ["r_1", "r_2", "r_3", "r", "r_4"]
    assert p.result == Ref("r_4")


def test_early_return_top_level() -> None:
    p = lower("    if x < 0:\n        return 0.0\n    return x * math.exp(-x)")
    assert p.bindings == ()
    assert p.result == Where(
        Compare("lt", Ref("x"), Literal(0, "int")),
        Literal(0.0, "float"),
        BinOp(
            "mul",
            Ref("x"),
            Call(
                "exp",
                (Call("astype", (Call("asarray", (UnaryOp("neg", Ref("x")),)), DType("float64"))),),
            ),
        ),
    )


def test_sequential_same_condition_first_return_wins() -> None:
    p = lower(
        "    if x < 0:\n        return 1.0\n    if x < 0:\n        return 2.0\n    return 3.0"
    )
    inner = Where(
        Compare("lt", Ref("x"), Literal(0, "int")), Literal(2.0, "float"), Literal(3.0, "float")
    )
    assert p.result == Where(
        Compare("lt", Ref("x"), Literal(0, "int")), Literal(1.0, "float"), inner
    )


def test_nested_both_return_with_outer_pending() -> None:
    p = lower(
        "    if x < -5:\n"
        "        return 9.0\n"
        "    if x > 1:\n"
        "        if x > 2:\n"
        "            return 1.0\n"
        "        else:\n"
        "            return 2.0\n"
        "    return 3.0"
    )
    inner = Where(
        Compare("gt", Ref("x"), Literal(2, "int")), Literal(1.0, "float"), Literal(2.0, "float")
    )
    middle = Where(Compare("gt", Ref("x"), Literal(1, "int")), inner, Literal(3.0, "float"))
    assert p.result == Where(
        Compare("lt", Ref("x"), UnaryOp("neg", Literal(5, "int"))), Literal(9.0, "float"), middle
    )


def test_then_return_with_inner_else_pending() -> None:
    p = lower(
        "    if x > 0:\n"
        "        return 1.0\n"
        "    else:\n"
        "        if x < -1:\n"
        "            return 2.0\n"
        "    return 3.0"
    )
    inner = Where(
        Compare("lt", Ref("x"), UnaryOp("neg", Literal(1, "int"))),
        Literal(2.0, "float"),
        Literal(3.0, "float"),
    )
    assert p.result == Where(
        Compare("gt", Ref("x"), Literal(0, "int")), Literal(1.0, "float"), inner
    )


def test_early_return_in_else_branch() -> None:
    p = lower("    if x < 0:\n        y = -x\n    else:\n        return 0.0\n    return y")
    assert p.result == Where(
        UnaryOp("not", Compare("lt", Ref("x"), Literal(0, "int"))),
        Literal(0.0, "float"),
        Ref("y_1"),
    )


def test_both_branches_return() -> None:
    p = lower("    if x < 0:\n        return 0.0\n    else:\n        return x\n")
    assert p.result == Where(
        Compare("lt", Ref("x"), Literal(0, "int")), Literal(0.0, "float"), Ref("x")
    )


def test_nested_if_with_returns() -> None:
    p = lower(
        "    if x > 0:\n"
        "        if x > 10:\n"
        "            return 1.0\n"
        "        return 2.0\n"
        "    return 3.0"
    )
    assert p.result == Where(
        Compare("gt", Ref("x"), Literal(0, "int")),
        Where(
            Compare("gt", Ref("x"), Literal(10, "int")),
            Literal(1.0, "float"),
            Literal(2.0, "float"),
        ),
        Literal(3.0, "float"),
    )


def test_pending_return_qualified_by_branch() -> None:
    # inner early return inside a branch that falls through must be
    # conditioned on the branch condition
    p = lower(
        "    if x > 0:\n"
        "        if x > 10:\n"
        "            return 1.0\n"
        "        y = 2.0\n"
        "    else:\n"
        "        y = 3.0\n"
        "    return y"
    )
    assert isinstance(p.result, Where)
    outer_cond = p.result.cond
    assert isinstance(outer_cond, Logical)
    assert outer_cond.op == "and"


def test_bare_truthiness_if_rejected() -> None:
    with pytest.raises(VectorizationError, match="truthiness"):
        lower("    if x:\n        return x\n    return 0.0")


def test_maybe_unbound_read_rejected() -> None:
    with pytest.raises(VectorizationError, match="may be unbound"):
        lower("    if x > 0:\n        z = 1.0\n    return z")


def test_maybe_unbound_not_read_is_fine() -> None:
    p = lower("    if x > 0:\n        z = 1.0\n    return x")
    assert p.result == Ref("x")
    # z_1 is dead; DCE (in _optimize) drops it later
    assert [b.name for b in p.bindings] == ["z_1"]


def test_not_all_paths_return_rejected() -> None:
    with pytest.raises(VectorizationError, match="paths return"):
        lower("    if x > 0:\n        return x")


def test_lambda_lowering() -> None:
    path = _TMPDIR / f"snippet_{next(_seq)}.py"
    path.write_text("subject = lambda x: x + 1.0\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lf = lower_function(extract_function(mod.subject))
    assert lf.program.result == BinOp("add", Ref("x"), Literal(1.0, "float"))
