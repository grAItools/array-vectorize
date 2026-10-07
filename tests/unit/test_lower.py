# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Lowering tests: AST -> IR shapes, straight-line + control flow."""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any, cast

import numpy as np
import pytest
import support

import array_vectorize
from array_vectorize import errors
from array_vectorize import ir
from array_vectorize import lower as lower_mod
from array_vectorize.frontend import extract

GLOBAL_K = 3
GLOBAL_ARR = np.asarray([10.0, 20.0])


def make_fn(body: str, extra_globals: str = "") -> Callable[..., Any]:
    mod = support.make_module(
        "import math\n"
        "from math import exp\n"
        "GLOBAL_K = 3\n"
        "GLOBAL_ARR = __import__('numpy').asarray([10.0, 20.0])\n"
        + extra_globals
        + "\n\n\ndef subject(x, y=2.0):\n"
        + body
        + "\n"
    )
    return cast(Callable[..., Any], mod.subject)


def lower(body: str) -> ir.Program:
    return lower_mod.lower_function(extract.extract_function(make_fn(body))).program


# ----------------------------------------------------------------- expressions


def test_const_and_binop() -> None:
    assert lower("    return x + 1") == ir.Program(
        ("x", "y"), (), ir.BinOp("add", ir.Ref("x"), ir.Literal(1, "int"))
    )


def test_arith_ops_map() -> None:
    p = lower("    return x - y * 2 / 3")
    assert p.result == ir.BinOp(
        "sub",
        ir.Ref("x"),
        ir.FuncCall(
            "_vec_arith",
            (
                ir.Ref("xp"),
                ir.Literal(4, "int"),
                ir.BinOp("mul", ir.Ref("y"), ir.Literal(2, "int")),
                ir.Literal(3, "int"),
            ),
        ),
    )


def test_pow_floordiv_mod_bitwise() -> None:
    assert lower("    return x ** 2").result == ir.BinOp("pow", ir.Ref("x"), ir.Literal(2, "int"))
    assert lower("    return x // 2").result == ir.BinOp(
        "floordiv", ir.Ref("x"), ir.Literal(2, "int")
    )
    assert lower("    return x % 2").result == ir.BinOp("mod", ir.Ref("x"), ir.Literal(2, "int"))
    assert lower("    return x & 1").result == ir.BinOp("and", ir.Ref("x"), ir.Literal(1, "int"))
    assert lower("    return x | 1").result == ir.BinOp("or", ir.Ref("x"), ir.Literal(1, "int"))
    assert lower("    return x ^ 1").result == ir.BinOp("xor", ir.Ref("x"), ir.Literal(1, "int"))
    assert lower("    return x << 1").result == ir.BinOp(
        "lshift", ir.Ref("x"), ir.Literal(1, "int")
    )
    assert lower("    return x >> 1").result == ir.BinOp(
        "rshift", ir.Ref("x"), ir.Literal(1, "int")
    )


def test_unary_ops() -> None:
    assert lower("    return -x").result == ir.UnaryOp("neg", ir.Ref("x"))
    assert lower("    return +x").result == ir.UnaryOp("pos", ir.Ref("x"))
    assert lower("    return ~x").result == ir.UnaryOp("invert", ir.Ref("x"))
    assert lower("    return not (x > 0)").result == ir.UnaryOp(
        "not", ir.Compare("gt", ir.Ref("x"), ir.Literal(0, "int"))
    )


def test_bool_literals_typed() -> None:
    # bool literal in arithmetic: the whole operation goes through the
    # runtime _vec_arith helper (bools need a numeric representation;
    # backends saturate or reject bool arithmetic)
    assert lower("    return x + True").result == ir.FuncCall(
        "_vec_arith", (ir.Ref("xp"), ir.Literal(1, "int"), ir.Ref("x"), ir.Literal(True, "bool"))
    )
    assert lower("    return 1.5").result == ir.Literal(1.5, "float")


def test_math_attribute_call() -> None:
    assert lower("    return math.sqrt(x)").result == ir.Call(
        "sqrt", (ir.Call("astype", (ir.Call("asarray", (ir.Ref("x"),)), ir.DType("float64"))),)
    )
    assert lower("    return math.atan2(x, y)").result == ir.Call(
        "atan2",
        (
            ir.Call("astype", (ir.Call("asarray", (ir.Ref("x"),)), ir.DType("float64"))),
            ir.Call("astype", (ir.Call("asarray", (ir.Ref("y"),)), ir.DType("float64"))),
        ),
    )
    assert lower("    return math.trunc(x)").result == ir.Call(
        "astype", (ir.Call("asarray", (ir.Ref("x"),)), ir.DType("int64"))
    )
    assert lower("    return math.pow(x, 2)").result == ir.Call(
        "pow",
        (
            ir.Call("astype", (ir.Call("asarray", (ir.Ref("x"),)), ir.DType("float64"))),
            ir.Literal(2.0, "float"),
        ),
    )


def test_from_math_import_call() -> None:
    assert lower("    return exp(x)").result == ir.Call(
        "exp", (ir.Call("astype", (ir.Call("asarray", (ir.Ref("x"),)), ir.DType("float64"))),)
    )


def test_math_constants() -> None:
    assert lower("    return math.pi").result == ir.Literal(math.pi, "float")
    assert lower("    return math.inf").result == ir.Literal(math.inf, "float")


def test_builtins() -> None:
    assert lower("    return abs(x)").result == ir.Call(
        "abs", (ir.Call("asarray", (ir.Ref("x"),)),)
    )
    assert lower("    return round(x)").result == ir.Call(
        "round", (ir.Call("asarray", (ir.Ref("x"),)),)
    )
    assert lower("    return min(x, y)").result == ir.FuncCall(
        "_vec_minmax",
        (
            ir.Ref("xp"),
            ir.Literal(True, "bool"),
            ir.Call("asarray", (ir.Ref("x"),)),
            ir.Call("asarray", (ir.Ref("y"),)),
        ),
    )
    assert lower("    return max(x, y, 2.0)").result == ir.FuncCall(
        "_vec_minmax",
        (
            ir.Ref("xp"),
            ir.Literal(False, "bool"),
            ir.Call("asarray", (ir.Ref("x"),)),
            ir.Call("asarray", (ir.Ref("y"),)),
            ir.Literal(2.0, "float"),
        ),
    )
    assert lower("    return int(x)").result == ir.Call(
        "astype", (ir.Call("asarray", (ir.Ref("x"),)), ir.DType("int64"))
    )
    assert lower("    return float(x)").result == ir.Call(
        "astype", (ir.Call("asarray", (ir.Ref("x"),)), ir.DType("float64"))
    )
    assert lower("    return bool(x)").result == ir.Call(
        "astype", (ir.Call("asarray", (ir.Ref("x"),)), ir.DType("bool"))
    )


def test_comparisons_and_chains() -> None:
    assert lower("    return x < y").result == ir.Compare("lt", ir.Ref("x"), ir.Ref("y"))
    assert lower("    return 0 < x < 1").result == ir.Logical(
        "and",
        (
            ir.Compare("lt", ir.Literal(0, "int"), ir.Ref("x")),
            ir.Compare("lt", ir.Ref("x"), ir.Literal(1, "int")),
        ),
    )


def test_boolop_bool_operands_use_logical() -> None:
    assert lower("    return (x < 0) and (y > 0)").result == ir.Logical(
        "and",
        (
            ir.Compare("lt", ir.Ref("x"), ir.Literal(0, "int")),
            ir.Compare("gt", ir.Ref("y"), ir.Literal(0, "int")),
        ),
    )


def test_boolop_numeric_value_select() -> None:
    assert lower("    return x and y").result == ir.Where(
        ir.Compare("ne", ir.Ref("x"), ir.Literal(0, "int")), ir.Ref("y"), ir.Ref("x")
    )
    assert lower("    return x or y").result == ir.Where(
        ir.Compare("ne", ir.Ref("x"), ir.Literal(0, "int")), ir.Ref("x"), ir.Ref("y")
    )


def test_boolop_numeric_chain_uses_temp() -> None:
    p = lower("    return x and y and x")
    assert len(p.bindings) == 1
    temp = p.bindings[0]
    assert support.binding(temp).name == "v_1"
    assert support.binding(temp).expr == ir.Where(
        ir.Compare("ne", ir.Ref("x"), ir.Literal(0, "int")), ir.Ref("y"), ir.Ref("x")
    )
    assert p.result == ir.Where(
        ir.Compare("ne", ir.Ref("v_1"), ir.Literal(0, "int")), ir.Ref("x"), ir.Ref("v_1")
    )


def test_ternary_coerces_numeric_condition() -> None:
    assert lower("    return x if x else y").result == ir.Where(
        ir.Compare("ne", ir.Ref("x"), ir.Literal(0, "int")), ir.Ref("x"), ir.Ref("y")
    )
    assert lower("    return x if x > 0 else y").result == ir.Where(
        ir.Compare("gt", ir.Ref("x"), ir.Literal(0, "int")), ir.Ref("x"), ir.Ref("y")
    )


# ---------------------------------------------------------------- statements


def test_assign_and_rebind() -> None:
    p = lower("    z = x * 2\n    z = z + 1\n    return z")
    assert p.bindings == (
        ir.Binding("z", ir.BinOp("mul", ir.Ref("x"), ir.Literal(2, "int"))),
        ir.Binding("z_1", ir.BinOp("add", ir.Ref("z"), ir.Literal(1, "int"))),
    )
    assert p.result == ir.Ref("z_1")


def test_augassign() -> None:
    p = lower("    z = x\n    z += 1\n    z *= 2\n    return z")
    assert p.bindings == (
        ir.Binding("z", ir.Ref("x")),
        ir.Binding("z_1", ir.BinOp("add", ir.Ref("z"), ir.Literal(1, "int"))),
        ir.Binding("z_2", ir.BinOp("mul", ir.Ref("z_1"), ir.Literal(2, "int"))),
    )
    assert p.result == ir.Ref("z_2")


def test_rebind_skips_user_names() -> None:
    p = lower("    x_1 = 0\n    x = x + 1\n    x = x + 1\n    return x + x_1")
    names = [support.binding(b).name for b in p.bindings]
    assert names[1:] == ["x_2", "x_3"]  # x_1 taken by the user variable


def test_closure_scalar_frozen() -> None:
    assert lower("    return x * GLOBAL_K").result == ir.BinOp(
        "mul", ir.Ref("x"), ir.Literal(3, "int")
    )


def test_closure_array_hidden_param() -> None:
    lf = lower_mod.lower_function(extract.extract_function(make_fn("    return x + GLOBAL_ARR")))
    assert lf.program.params == ("x", "y", "GLOBAL_ARR")
    assert lf.program.result == ir.BinOp("add", ir.Ref("x"), ir.Ref("GLOBAL_ARR"))
    assert len(lf.hidden_params) == 1
    name, default = lf.hidden_params[0]
    assert name == "GLOBAL_ARR"
    assert np.array_equal(default, [10.0, 20.0])


def test_reserved_param_name_kept_namespace_renamed() -> None:
    # a parameter named 'xp' keeps its name (keyword calls work); the
    # generated namespace variable renames itself instead
    mod = support.make_module("def subject(xp):\n    return xp + 1\n")
    lf = lower_mod.lower_function(extract.extract_function(mod.subject))
    assert lf.param_names == ["xp"]
    assert lf.namespace_var == "xp_1"
    assert lf.program.result == ir.BinOp("add", ir.Ref("xp"), ir.Literal(1, "int"))


# ----------------------------------------------------------------- rejections


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
    with pytest.raises(errors.VectorizationError):
        lower(body)


def test_reference_before_assignment() -> None:
    with pytest.raises(errors.VectorizationError, match="before assignment"):
        lower("    z = z + 1\n    return z")


def test_closure_scalar_read_after_local_assign_rejected() -> None:
    # GLOBAL_K is assigned locally -> reads before the assignment are unbound
    with pytest.raises(errors.VectorizationError, match="before assignment"):
        lower("    z = GLOBAL_K + 1\n    GLOBAL_K = 2\n    return z")


def test_calling_variable_rejected() -> None:
    with pytest.raises(errors.VectorizationError, match="calling variable"):
        lower("    f = x\n    return f(x)")


# ------------------------------------------------------------------ if/else


def test_if_else_merge() -> None:
    p = lower("    if x > 0:\n        r = x\n    else:\n        r = 0.0\n    return r")
    assert [support.binding(b).name for b in p.bindings] == ["r_1", "r_2", "r"]
    assert p.bindings[0] == ir.Binding("r_1", ir.Ref("x"))
    assert p.bindings[1] == ir.Binding("r_2", ir.Literal(0.0, "float"))
    assert p.bindings[2] == ir.Binding(
        "r",
        ir.Where(ir.Compare("gt", ir.Ref("x"), ir.Literal(0, "int")), ir.Ref("r_1"), ir.Ref("r_2")),
    )
    assert p.result == ir.Ref("r")


def test_if_else_merge_with_prior_binding() -> None:
    p = lower("    r = 0.0\n    if x > 0:\n        r = x\n    else:\n        r = 1.0\n    return r")
    assert [support.binding(b).name for b in p.bindings] == ["r", "r_1", "r_2", "r_3"]
    assert p.result == ir.Ref("r_3")


def test_if_without_else_and_prior_binding() -> None:
    p = lower("    r = 0.0\n    if x > 0:\n        r = x\n    return r")
    # then binds r_1; else keeps r; merge where(cond, r_1, r)
    assert [support.binding(b).name for b in p.bindings] == ["r", "r_1", "r_2"]
    assert support.binding(p.bindings[2]).expr == ir.Where(
        ir.Compare("gt", ir.Ref("x"), ir.Literal(0, "int")), ir.Ref("r_1"), ir.Ref("r")
    )
    assert p.result == ir.Ref("r_2")


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
    names = [support.binding(b).name for b in p.bindings]
    assert names == ["r_1", "r_2", "r_3", "r", "r_4"]
    assert p.result == ir.Ref("r_4")


def test_early_return_top_level() -> None:
    p = lower("    if x < 0:\n        return 0.0\n    return x * math.exp(-x)")
    assert p.bindings == ()
    assert p.result == ir.Where(
        ir.Compare("lt", ir.Ref("x"), ir.Literal(0, "int")),
        ir.Literal(0.0, "float"),
        ir.BinOp(
            "mul",
            ir.Ref("x"),
            ir.Call(
                "exp",
                (
                    ir.Call(
                        "astype",
                        (
                            ir.Call("asarray", (ir.UnaryOp("neg", ir.Ref("x")),)),
                            ir.DType("float64"),
                        ),
                    ),
                ),
            ),
        ),
    )


def test_sequential_same_condition_first_return_wins() -> None:
    p = lower(
        "    if x < 0:\n        return 1.0\n    if x < 0:\n        return 2.0\n    return 3.0"
    )
    inner = ir.Where(
        ir.Compare("lt", ir.Ref("x"), ir.Literal(0, "int")),
        ir.Literal(2.0, "float"),
        ir.Literal(3.0, "float"),
    )
    assert p.result == ir.Where(
        ir.Compare("lt", ir.Ref("x"), ir.Literal(0, "int")), ir.Literal(1.0, "float"), inner
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
    inner = ir.Where(
        ir.Compare("gt", ir.Ref("x"), ir.Literal(2, "int")),
        ir.Literal(1.0, "float"),
        ir.Literal(2.0, "float"),
    )
    middle = ir.Where(
        ir.Compare("gt", ir.Ref("x"), ir.Literal(1, "int")), inner, ir.Literal(3.0, "float")
    )
    assert p.result == ir.Where(
        ir.Compare("lt", ir.Ref("x"), ir.UnaryOp("neg", ir.Literal(5, "int"))),
        ir.Literal(9.0, "float"),
        middle,
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
    inner = ir.Where(
        ir.Compare("lt", ir.Ref("x"), ir.UnaryOp("neg", ir.Literal(1, "int"))),
        ir.Literal(2.0, "float"),
        ir.Literal(3.0, "float"),
    )
    assert p.result == ir.Where(
        ir.Compare("gt", ir.Ref("x"), ir.Literal(0, "int")), ir.Literal(1.0, "float"), inner
    )


def test_early_return_in_else_branch() -> None:
    p = lower("    if x < 0:\n        y = -x\n    else:\n        return 0.0\n    return y")
    assert p.result == ir.Where(
        ir.UnaryOp("not", ir.Compare("lt", ir.Ref("x"), ir.Literal(0, "int"))),
        ir.Literal(0.0, "float"),
        ir.Ref("y_1"),
    )


def test_both_branches_return() -> None:
    p = lower("    if x < 0:\n        return 0.0\n    else:\n        return x\n")
    assert p.result == ir.Where(
        ir.Compare("lt", ir.Ref("x"), ir.Literal(0, "int")), ir.Literal(0.0, "float"), ir.Ref("x")
    )


def test_nested_if_with_returns() -> None:
    p = lower(
        "    if x > 0:\n"
        "        if x > 10:\n"
        "            return 1.0\n"
        "        return 2.0\n"
        "    return 3.0"
    )
    assert p.result == ir.Where(
        ir.Compare("gt", ir.Ref("x"), ir.Literal(0, "int")),
        ir.Where(
            ir.Compare("gt", ir.Ref("x"), ir.Literal(10, "int")),
            ir.Literal(1.0, "float"),
            ir.Literal(2.0, "float"),
        ),
        ir.Literal(3.0, "float"),
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
    assert isinstance(p.result, ir.Where)
    outer_cond = p.result.cond
    assert isinstance(outer_cond, ir.Logical)
    assert outer_cond.op == "and"


def test_bare_truthiness_if_rejected() -> None:
    with pytest.raises(errors.VectorizationError, match="truthiness"):
        lower("    if x:\n        return x\n    return 0.0")


def test_maybe_unbound_read_rejected() -> None:
    with pytest.raises(errors.VectorizationError, match="may be unbound"):
        lower("    if x > 0:\n        z = 1.0\n    return z")


def test_maybe_unbound_not_read_is_fine() -> None:
    p = lower("    if x > 0:\n        z = 1.0\n    return x")
    assert p.result == ir.Ref("x")
    # z_1 is dead; DCE (in _optimize) drops it later
    assert [support.binding(b).name for b in p.bindings] == ["z_1"]


def test_not_all_paths_return_rejected() -> None:
    with pytest.raises(errors.VectorizationError, match="paths return"):
        lower("    if x > 0:\n        return x")


def test_lambda_lowering() -> None:
    mod = support.make_module("subject = lambda x: x + 1.0\n")
    lf = lower_mod.lower_function(extract.extract_function(mod.subject))
    assert lf.program.result == ir.BinOp("add", ir.Ref("x"), ir.Literal(1.0, "float"))


def test_kinds_restore_revokes_post_snapshot_facts() -> None:
    # the loop-var "int" insert must be re-insertable on every fixed-point
    # pass: restore REPLACES the kind facts (old dict-reassignment
    # semantics), it must not leave a present-None entry that blocks
    # setdefault
    from array_vectorize.lower import kinds

    k = kinds.Kinds()
    snap = k.snapshot_facts()  # loop name not yet established
    k.setdefault_kind("i", "int")  # post-snapshot insert (loop-var default)
    k.restore_facts(snap)  # widening re-lower: state rolls back
    assert k.kind("i") is None  # fact gone ...
    k.setdefault_kind("i", "int")  # ... so the insert fires again
    assert k.kind("i") == "int"


def test_kinds_restore_preserves_present_none_kind() -> None:
    # a pre-bound loop var with unknown kind establishes a present-None
    # fact; the restore must keep both presence and value so the later
    # "int" default stays blocked (old dict setdefault semantics)
    from array_vectorize.lower import kinds

    k = kinds.Kinds()
    k.set_kind("i", None)  # phi with unknown kind
    snap = k.snapshot_facts()
    k.set_kind("i", "int")  # discarded pass wrote a kind
    k.restore_facts(snap)
    assert k.kind("i") is None
    k.setdefault_kind("i", "int")  # blocked: presence survived the restore
    assert k.kind("i") is None


# ------------------------------------------------------------ lower errors


def test_calling_math_module_rejected() -> None:
    with pytest.raises(errors.VectorizationError, match="math module"):
        array_vectorize.vectorize(make_fn("    return math(x)"))


def test_math_const_called_rejected() -> None:
    with pytest.raises(errors.VectorizationError, match="cannot be called"):
        array_vectorize.vectorize(make_fn("    return math.pi(x)"))


def test_calling_shadowed_builtin_rejected() -> None:
    with pytest.raises(errors.VectorizationError, match="calling variable"):
        array_vectorize.vectorize(make_fn("    abs = x\n    return abs(x)"))


def test_not_all_paths_return_via_branch_fallthrough() -> None:
    with pytest.raises(errors.VectorizationError, match="paths return"):
        array_vectorize.vectorize(
            make_fn("    if x > 0:\n        return 1.0\n    if x < -1:\n        return 2.0\n")
        )


def test_maybe_unbound_after_nested_branch() -> None:
    with pytest.raises(errors.VectorizationError, match="may be unbound"):
        array_vectorize.vectorize(
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


@pytest.mark.parametrize("op", ["%", "//"])
def test_unfoldable_boolean_literals_keep_runtime_arithmetic(op: str) -> None:
    p = lower(f"    return True {op} False")
    assert isinstance(p.result, ir.FuncCall)
    assert p.result.fn == "_vec_arith"
