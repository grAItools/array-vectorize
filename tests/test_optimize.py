"""Optimizer tests: const-fold, DCE, CSE (plan §8)."""

from __future__ import annotations

import math

from vectorizer._ir import (
    Binding,
    BinOp,
    Call,
    Compare,
    Literal,
    Program,
    Ref,
    SSAEnv,
    UnaryOp,
    Where,
)
from vectorizer._optimize import const_fold, cse, dce, optimize


def lit(v: float | int, kind: str = "auto") -> Literal:
    if kind == "auto":
        kind = "bool" if isinstance(v, bool) else "int" if isinstance(v, int) else "float"
    return Literal(v, kind)


# ---------------------------------------------------------------- const-fold


def test_fold_arithmetic() -> None:
    folded = const_fold(Program((), (), BinOp("add", lit(2), lit(3))))
    assert folded.result == lit(5)


def test_fold_nested() -> None:
    expr = BinOp("mul", BinOp("add", lit(2), lit(3)), lit(4))
    assert const_fold(Program((), (), expr)).result == lit(20)


def test_fold_div_makes_float() -> None:
    assert const_fold(Program((), (), BinOp("div", lit(6), lit(3)))).result == lit(2.0, "float")


def test_fold_div_by_zero_skipped() -> None:
    expr = BinOp("div", lit(1.0), lit(0.0))
    assert const_fold(Program((), (), expr)).result == expr


def test_fold_mod_by_zero_skipped() -> None:
    expr = BinOp("mod", lit(1.0), lit(0.0))
    assert const_fold(Program((), (), expr)).result == expr


def test_fold_pow_skipped() -> None:
    expr = BinOp("pow", lit(2), lit(3))
    assert const_fold(Program((), (), expr)).result == expr


def test_fold_int64_overflow_skipped() -> None:
    expr = BinOp("mul", lit(2**62), lit(4))
    assert const_fold(Program((), (), expr)).result == expr
    expr = UnaryOp("neg", lit(-(2**63)))
    assert const_fold(Program((), (), expr)).result == expr


def test_fold_bitwise_on_bools_keeps_bool() -> None:
    assert const_fold(Program((), (), BinOp("and", lit(True), lit(False)))).result == lit(
        False, "bool"
    )


def test_fold_unary() -> None:
    assert const_fold(Program((), (), UnaryOp("neg", lit(5)))).result == lit(-5)
    assert const_fold(Program((), (), UnaryOp("invert", lit(5)))).result == lit(-6)
    assert const_fold(Program((), (), UnaryOp("pos", lit(5)))).result == lit(5)


def test_fold_not_is_exact_for_nan() -> None:
    assert const_fold(Program((), (), UnaryOp("not", lit(math.nan, "float")))).result == lit(
        False, "bool"
    )
    assert const_fold(Program((), (), UnaryOp("not", lit(0.0, "float")))).result == lit(
        True, "bool"
    )


def test_fold_does_not_touch_calls() -> None:
    expr = Call("sqrt", (lit(4),))
    assert const_fold(Program((), (), expr)).result == expr


# ----------------------------------------------------------------------- DCE


def test_dce_drops_unreachable_bindings() -> None:
    p = Program(
        ("x",),
        (
            Binding("a", Call("sqrt", (Ref("x"),))),
            Binding("dead", Call("exp", (Ref("x"),))),
        ),
        Ref("a"),
    )
    assert dce(p).bindings == (Binding("a", Call("sqrt", (Ref("x"),))),)


def test_dce_keeps_chains() -> None:
    p = Program(
        ("x",),
        (
            Binding("a", Call("sqrt", (Ref("x"),))),
            Binding("b", Call("exp", (Ref("a"),))),
            Binding("dead", Ref("x")),
        ),
        Ref("b"),
    )
    assert [b.name for b in dce(p).bindings] == ["a", "b"]


def test_dce_single_reverse_pass_handles_late_uses() -> None:
    p = Program(
        ("x",),
        (
            Binding("a", Ref("x")),
            Binding("b", Call("sqrt", (Ref("a"),))),
        ),
        Ref("b"),
    )
    assert [b.name for b in dce(p).bindings] == ["a", "b"]


# ----------------------------------------------------------------------- CSE


def test_cse_hoists_duplicated_call() -> None:
    inner = Call("sqrt", (Ref("x"),))
    p = Program(
        ("x",),
        (Binding("a", BinOp("add", inner, inner)),),
        BinOp("mul", inner, Ref("a")),
    )
    out = cse(p, SSAEnv(set()))
    assert out.bindings[0].name == "t_1"
    assert out.bindings[0].expr == inner
    assert out.bindings[1] == Binding("a", BinOp("add", Ref("t_1"), Ref("t_1")))
    assert out.result == BinOp("mul", Ref("t_1"), Ref("a"))


def test_cse_no_duplicates_unchanged() -> None:
    p = Program(
        ("x",),
        (Binding("a", Call("sqrt", (Ref("x"),))),),
        Call("sqrt", (Ref("a"),)),
    )
    assert cse(p, SSAEnv(set())) == p


def test_cse_where_eligible_literals_not() -> None:
    where = Where(Compare("gt", Ref("x"), lit(0)), lit(1.0, "float"), lit(2.0, "float"))
    p = Program(("x",), (), BinOp("add", where, where))
    out = cse(p, SSAEnv(set()))
    assert out.bindings == (Binding("t_1", where),)
    assert out.result == BinOp("add", Ref("t_1"), Ref("t_1"))


def test_cse_nested_fixpoint() -> None:
    inner = Call("sqrt", (Ref("x"),))
    outer1 = Where(Compare("gt", inner, lit(0)), inner, lit(0.0, "float"))
    p = Program(("x",), (), BinOp("add", outer1, outer1))
    out = cse(p, SSAEnv(set()))
    names = [b.name for b in out.bindings]
    # inner temp created first, then (next pass) the rebuilt outer
    assert names == ["t_1", "t_2"]
    assert out.result == BinOp("add", Ref("t_2"), Ref("t_2"))


def test_cse_temp_inserted_before_first_user() -> None:
    inner = Call("sqrt", (Ref("x"),))
    p = Program(
        ("x",),
        (
            Binding("a", Ref("x")),
            Binding("b", BinOp("add", inner, inner)),
        ),
        Ref("b"),
    )
    out = cse(p, SSAEnv(set()))
    assert [b.name for b in out.bindings] == ["a", "t_1", "b"]


# ----------------------------------------------------------------- composite


def test_optimize_pipeline() -> None:
    # (x + (2*3)) duplicated, dead binding present
    sub = BinOp("add", Ref("x"), BinOp("mul", lit(2), lit(3)))
    p = Program(
        ("x",),
        (
            Binding("dead", Call("exp", (Ref("x"),))),
            Binding("a", Call("abs", (sub,))),
        ),
        BinOp("add", Call("abs", (sub,)), Ref("a")),
    )
    out = optimize(p, {"x", "dead", "a"})
    folded_sub = BinOp("add", Ref("x"), lit(6))
    assert out.bindings[0].name == "t_1"
    assert out.bindings[0].expr == Call("abs", (folded_sub,))
    assert out.bindings[1] == Binding("a", Ref("t_1"))
    assert out.result == BinOp("add", Ref("t_1"), Ref("a"))
