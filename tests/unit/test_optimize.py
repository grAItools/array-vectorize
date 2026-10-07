"""Optimizer tests: const-fold, DCE, CSE."""

from __future__ import annotations

import math
from typing import Literal

import pytest
import support

from array_vectorize import ir
from array_vectorize import optimize as optimize_mod

# optimize's documented submodule/function name collision (the NOTE in
# optimize/__init__.py): the from-import is the supported spelling
from array_vectorize.optimize import cse  # cleanporter: ignore[CP002] name collision
from array_vectorize.optimize import dce  # cleanporter: ignore[CP002] name collision
from array_vectorize.optimize import dce as dce_pass  # cleanporter: ignore[CP002] name collision


def lit(v: float | int, kind: ir.Kind | Literal["auto"] = "auto") -> ir.Literal:
    if kind == "auto":
        kind = "bool" if isinstance(v, bool) else "int" if isinstance(v, int) else "float"
    return ir.Literal(v, kind)


# ---------------------------------------------------------------- const-fold


def test_fold_arithmetic() -> None:
    folded = optimize_mod.const_fold(ir.Program((), (), ir.BinOp("add", lit(2), lit(3))))
    assert folded.result == lit(5)


def test_fold_nested() -> None:
    expr = ir.BinOp("mul", ir.BinOp("add", lit(2), lit(3)), lit(4))
    assert optimize_mod.const_fold(ir.Program((), (), expr)).result == lit(20)


def test_fold_div_makes_float() -> None:
    assert optimize_mod.const_fold(
        ir.Program((), (), ir.BinOp("div", lit(6), lit(3)))
    ).result == lit(2.0, "float")


def test_fold_div_by_zero_skipped() -> None:
    expr = ir.BinOp("div", lit(1.0), lit(0.0))
    assert optimize_mod.const_fold(ir.Program((), (), expr)).result == expr


def test_fold_mod_by_zero_skipped() -> None:
    expr = ir.BinOp("mod", lit(1.0), lit(0.0))
    assert optimize_mod.const_fold(ir.Program((), (), expr)).result == expr


def test_fold_pow_skipped() -> None:
    expr = ir.BinOp("pow", lit(2), lit(3))
    assert optimize_mod.const_fold(ir.Program((), (), expr)).result == expr


def test_fold_int64_overflow_skipped() -> None:
    expr: ir.Node = ir.BinOp("mul", lit(2**62), lit(4))
    assert optimize_mod.const_fold(ir.Program((), (), expr)).result == expr
    expr = ir.UnaryOp("neg", lit(-(2**63)))
    assert optimize_mod.const_fold(ir.Program((), (), expr)).result == expr


def test_fold_bitwise_on_bools_keeps_bool() -> None:
    assert optimize_mod.const_fold(
        ir.Program((), (), ir.BinOp("and", lit(True), lit(False)))
    ).result == lit(False, "bool")


def test_fold_unary() -> None:
    assert optimize_mod.const_fold(ir.Program((), (), ir.UnaryOp("neg", lit(5)))).result == lit(-5)
    assert optimize_mod.const_fold(ir.Program((), (), ir.UnaryOp("invert", lit(5)))).result == lit(
        -6
    )
    assert optimize_mod.const_fold(ir.Program((), (), ir.UnaryOp("pos", lit(5)))).result == lit(5)


def test_fold_not_is_exact_for_nan() -> None:
    assert optimize_mod.const_fold(
        ir.Program((), (), ir.UnaryOp("not", lit(math.nan, "float")))
    ).result == lit(False, "bool")
    assert optimize_mod.const_fold(
        ir.Program((), (), ir.UnaryOp("not", lit(0.0, "float")))
    ).result == lit(True, "bool")


def test_fold_pure_calls_over_literals() -> None:
    # pure xp functions with literal args fold (keeps plain scalars out of
    # generated code: strict backends require arrays)
    assert optimize_mod.const_fold(ir.Program((), (), ir.Call("sqrt", (lit(4),)))).result == lit(
        2.0
    )
    expr = ir.Call("sqrt", (lit(-1),))  # math.sqrt(-1) raises: no fold
    assert optimize_mod.const_fold(ir.Program((), (), expr)).result == expr
    expr = ir.Call("where", (lit(1), lit(2), lit(3)))  # not in the fold table
    assert optimize_mod.const_fold(ir.Program((), (), expr)).result == expr
    expr = ir.Call("sqrt", (ir.Ref("x"),))  # non-literal arg: no fold
    assert optimize_mod.const_fold(ir.Program((), (), expr)).result == expr


# ----------------------------------------------------------------------- DCE


def test_dce_drops_unreachable_bindings() -> None:
    p = ir.Program(
        ("x",),
        (
            ir.Binding("a", ir.Call("sqrt", (ir.Ref("x"),))),
            ir.Binding("dead", ir.Call("exp", (ir.Ref("x"),))),
        ),
        ir.Ref("a"),
    )
    assert dce(p).bindings == (ir.Binding("a", ir.Call("sqrt", (ir.Ref("x"),))),)


def test_dce_keeps_chains() -> None:
    p = ir.Program(
        ("x",),
        (
            ir.Binding("a", ir.Call("sqrt", (ir.Ref("x"),))),
            ir.Binding("b", ir.Call("exp", (ir.Ref("a"),))),
            ir.Binding("dead", ir.Ref("x")),
        ),
        ir.Ref("b"),
    )
    assert [support.binding(b).name for b in dce(p).bindings] == ["a", "b"]


def test_dce_single_reverse_pass_handles_late_uses() -> None:
    p = ir.Program(
        ("x",),
        (
            ir.Binding("a", ir.Ref("x")),
            ir.Binding("b", ir.Call("sqrt", (ir.Ref("a"),))),
        ),
        ir.Ref("b"),
    )
    assert [support.binding(b).name for b in dce(p).bindings] == ["a", "b"]


# ----------------------------------------------------------------------- CSE


def test_cse_hoists_duplicated_call() -> None:
    inner = ir.Call("sqrt", (ir.Ref("x"),))
    p = ir.Program(
        ("x",),
        (ir.Binding("a", ir.BinOp("add", inner, inner)),),
        ir.BinOp("mul", inner, ir.Ref("a")),
    )
    out = cse(p, ir.SSAEnv(set()))
    assert support.binding(out.bindings[0]).name == "t_1"
    assert support.binding(out.bindings[0]).expr == inner
    assert out.bindings[1] == ir.Binding("a", ir.BinOp("add", ir.Ref("t_1"), ir.Ref("t_1")))
    assert out.result == ir.BinOp("mul", ir.Ref("t_1"), ir.Ref("a"))


def test_cse_no_duplicates_unchanged() -> None:
    p = ir.Program(
        ("x",),
        (ir.Binding("a", ir.Call("sqrt", (ir.Ref("x"),))),),
        ir.Call("sqrt", (ir.Ref("a"),)),
    )
    assert cse(p, ir.SSAEnv(set())) == p


def test_cse_where_eligible_literals_not() -> None:
    where = ir.Where(ir.Compare("gt", ir.Ref("x"), lit(0)), lit(1.0, "float"), lit(2.0, "float"))
    p = ir.Program(("x",), (), ir.BinOp("add", where, where))
    out = cse(p, ir.SSAEnv(set()))
    assert out.bindings == (ir.Binding("t_1", where),)
    assert out.result == ir.BinOp("add", ir.Ref("t_1"), ir.Ref("t_1"))


def test_cse_nested_fixpoint() -> None:
    inner = ir.Call("sqrt", (ir.Ref("x"),))
    outer1 = ir.Where(ir.Compare("gt", inner, lit(0)), inner, lit(0.0, "float"))
    p = ir.Program(("x",), (), ir.BinOp("add", outer1, outer1))
    out = cse(p, ir.SSAEnv(set()))
    names = [support.binding(b).name for b in out.bindings]
    # inner temp created first, then (next pass) the rebuilt outer
    assert names == ["t_1", "t_2"]
    assert out.result == ir.BinOp("add", ir.Ref("t_2"), ir.Ref("t_2"))


def test_cse_temp_inserted_before_first_user() -> None:
    inner = ir.Call("sqrt", (ir.Ref("x"),))
    p = ir.Program(
        ("x",),
        (
            ir.Binding("a", ir.Ref("x")),
            ir.Binding("b", ir.BinOp("add", inner, inner)),
        ),
        ir.Ref("b"),
    )
    out = cse(p, ir.SSAEnv(set()))
    assert [support.binding(b).name for b in out.bindings] == ["a", "t_1", "b"]


# ----------------------------------------------------------------- composite


def test_optimize_pipeline() -> None:
    # (x + (2*3)) duplicated, dead binding present
    sub = ir.BinOp("add", ir.Ref("x"), ir.BinOp("mul", lit(2), lit(3)))
    p = ir.Program(
        ("x",),
        (
            ir.Binding("dead", ir.Call("exp", (ir.Ref("x"),))),
            ir.Binding("a", ir.Call("abs", (sub,))),
        ),
        ir.BinOp("add", ir.Call("abs", (sub,)), ir.Ref("a")),
    )
    out = optimize_mod.optimize(p, {"x", "dead", "a"})
    folded_sub = ir.BinOp("add", ir.Ref("x"), lit(6))
    assert support.binding(out.bindings[0]).name == "t_1"
    assert support.binding(out.bindings[0]).expr == ir.Call("abs", (folded_sub,))
    assert out.bindings[1] == ir.Binding("a", ir.Ref("t_1"))
    assert out.result == ir.BinOp("add", ir.Ref("t_1"), ir.Ref("a"))


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
    out = optimize_mod.const_fold(ir.Program((), (), ir.BinOp(op, lit(a), lit(b))))
    assert out.result == lit(expected)


def test_fold_unary_ops_all() -> None:
    assert optimize_mod.const_fold(ir.Program((), (), ir.UnaryOp("invert", lit(0)))).result == lit(
        -1
    )
    assert optimize_mod.const_fold(ir.Program((), (), ir.UnaryOp("pos", lit(-2)))).result == lit(-2)
    assert optimize_mod.const_fold(ir.Program((), (), ir.UnaryOp("neg", lit(2.5)))).result == lit(
        -2.5
    )
    assert optimize_mod.const_fold(ir.Program((), (), ir.UnaryOp("not", lit(True)))).result == lit(
        False
    )


def test_fold_skips_bad_ops() -> None:
    # float bitwise -> TypeError at runtime -> no fold
    expr: ir.Node = ir.BinOp("and", lit(1.5), lit(2.5))
    assert optimize_mod.const_fold(ir.Program((), (), expr)).result == expr
    # unary invert on float -> no fold
    expr = ir.UnaryOp("invert", lit(1.5))
    assert optimize_mod.const_fold(ir.Program((), (), expr)).result == expr
    # unknown unary op passes through
    expr = ir.UnaryOp("weird", lit(1))
    assert optimize_mod.const_fold(ir.Program((), (), expr)).result == expr
    # floordiv by zero
    expr = ir.BinOp("floordiv", lit(1), lit(0))
    assert optimize_mod.const_fold(ir.Program((), (), expr)).result == expr


def test_fold_zero_times_infinity() -> None:
    # inf * 0 -> nan: exact IEEE fold
    out = optimize_mod.const_fold(ir.Program((), (), ir.BinOp("mul", lit(math.inf), lit(0.0))))
    assert isinstance(out.result, ir.Literal)
    assert math.isnan(out.result.value)


def test_dce_keeps_result_only_binding() -> None:
    p = ir.Program(("x",), (ir.Binding("a", ir.Call("abs", (ir.Ref("x"),))),), ir.Ref("a"))
    assert dce_pass(p).bindings == p.bindings


def test_cse_stops_at_fixpoint() -> None:
    inner = ir.Call("sqrt", (ir.Ref("x"),))
    p = ir.Program(("x",), (), ir.BinOp("add", inner, inner))
    out = cse(p, ir.SSAEnv(set()))
    assert out.bindings == (ir.Binding("t_1", inner),)
    assert out.result == ir.BinOp("add", ir.Ref("t_1"), ir.Ref("t_1"))


def test_optimize_no_user_names() -> None:
    p = ir.Program(("x",), (ir.Binding("a", ir.Ref("x")),), ir.Ref("a"))
    assert optimize_mod.optimize(p).result == ir.Ref("a")
