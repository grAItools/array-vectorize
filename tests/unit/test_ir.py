# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Tests for the IR: node equality/hashability, is_bool lattice, SSA naming."""

from __future__ import annotations

import math

from array_vectorize import ir


class TestNodeBasics:
    def test_equality_is_structural(self) -> None:
        a = ir.BinOp("add", ir.Literal(1, "int"), ir.Ref("x"))
        b = ir.BinOp("add", ir.Literal(1, "int"), ir.Ref("x"))
        c = ir.BinOp("add", ir.Literal(2, "int"), ir.Ref("x"))
        assert a == b
        assert hash(a) == hash(b)
        assert a != c

    def test_inf_nan_literals_hashable(self) -> None:
        assert hash(ir.Literal(math.inf, "float")) == hash(ir.Literal(math.inf, "float"))
        assert hash(ir.Literal(math.nan, "float")) == hash(ir.Literal(math.nan, "float"))

    def test_nested_nodes_hashable(self) -> None:
        p = ir.Program(
            ("x",),
            (ir.Binding("v", ir.Call("sqrt", (ir.Ref("x"),))),),
            ir.Where(
                ir.Compare("lt", ir.Ref("x"), ir.Literal(0, "int")),
                ir.Literal(0.0, "float"),
                ir.Ref("v"),
            ),
        )
        assert hash(p) == hash(
            ir.Program(
                ("x",),
                (ir.Binding("v", ir.Call("sqrt", (ir.Ref("x"),))),),
                ir.Where(
                    ir.Compare("lt", ir.Ref("x"), ir.Literal(0, "int")),
                    ir.Literal(0.0, "float"),
                    ir.Ref("v"),
                ),
            )
        )

    def test_bool_is_not_int_literal(self) -> None:
        assert ir.Literal(True, "bool") != ir.Literal(1, "int")


class TestIsBool:
    def test_bool_nodes(self) -> None:
        assert ir.is_bool(ir.Compare("lt", ir.Ref("x"), ir.Literal(0, "int")))
        assert ir.is_bool(ir.UnaryOp("not", ir.Ref("x")))
        assert ir.is_bool(ir.Logical("and", (ir.Ref("a"), ir.Ref("b"))))
        assert ir.is_bool(ir.Literal(True, "bool"))
        assert ir.is_bool(ir.Call("isnan", (ir.Ref("x"),)))
        assert ir.is_bool(ir.Call("isinf", (ir.Ref("x"),)))
        assert ir.is_bool(ir.Call("isfinite", (ir.Ref("x"),)))

    def test_non_bool_nodes(self) -> None:
        assert not ir.is_bool(ir.Ref("x"))
        assert not ir.is_bool(ir.Literal(1, "int"))
        assert not ir.is_bool(ir.Literal(1.0, "float"))
        assert not ir.is_bool(ir.BinOp("add", ir.Ref("x"), ir.Ref("y")))
        assert not ir.is_bool(ir.UnaryOp("neg", ir.Ref("x")))
        assert not ir.is_bool(ir.Call("sqrt", (ir.Ref("x"),)))
        assert not ir.is_bool(ir.Where(ir.Ref("c"), ir.Ref("a"), ir.Ref("b")))
        assert not ir.is_bool(ir.DType("int64"))


class TestSSAEnv:
    def test_first_binding_keeps_name(self) -> None:
        env = ir.SSAEnv(["x"])
        assert env.bind("y") == "y"

    def test_rebind_picks_smallest_free_suffix(self) -> None:
        env = ir.SSAEnv(["x"])
        env.bind("x")
        assert env.bind("x") == "x_1"
        assert env.bind("x") == "x_2"

    def test_rebind_skips_user_names(self) -> None:
        # user has variables x and x_1: rebinding x must skip x_1
        env = ir.SSAEnv(["x", "x_1"])
        env.bind("x")
        assert env.bind("x") == "x_2"

    def test_reserved_namespace_name_mangled(self) -> None:
        # 'xp' is no longer reserved (the namespace var is renamed instead);
        # names the generated code cannot rename are still mangled
        env = ir.SSAEnv(["xp"])
        assert env.bind("xp") == "xp"

    def test_reserved_hasattr_mangled(self) -> None:
        env = ir.SSAEnv(["array_namespace"])
        assert env.bind("array_namespace") == "array_namespace_"
        env = ir.SSAEnv(["hasattr"])
        assert env.bind("hasattr") == "hasattr_"

    def test_mangle_skips_taken_mangled_name(self) -> None:
        # user has both array_namespace and array_namespace_ variables
        env = ir.SSAEnv(["array_namespace", "array_namespace_"])
        assert env.bind("array_namespace") == "array_namespace_1"

    def test_fresh_temp_skips_user_names(self) -> None:
        env = ir.SSAEnv(["t_1"])
        assert env.fresh_temp("t") == "t_2"

    def test_fresh_temp_skips_emitted(self) -> None:
        env = ir.SSAEnv([])
        env.fresh_temp("t")
        assert env.fresh_temp("t") == "t_2"

    def test_reserve_blocks_allocation(self) -> None:
        env = ir.SSAEnv(["g"])
        env.reserve("g_vec")
        env.bind("g")
        # g_vec is reserved and must never be handed out as a binding name
        for _ in range(3):
            name = env.bind("g")
            assert "g_vec" not in name
