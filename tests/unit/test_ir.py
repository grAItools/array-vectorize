"""Tests for the IR: node equality/hashability, is_bool lattice, SSA naming."""

from __future__ import annotations

import math

from array_vectorize.ir import (
    Binding,
    BinOp,
    Call,
    Compare,
    DType,
    Literal,
    Logical,
    Program,
    Ref,
    SSAEnv,
    UnaryOp,
    Where,
    is_bool,
)


class TestNodeBasics:
    def test_equality_is_structural(self) -> None:
        a = BinOp("add", Literal(1, "int"), Ref("x"))
        b = BinOp("add", Literal(1, "int"), Ref("x"))
        c = BinOp("add", Literal(2, "int"), Ref("x"))
        assert a == b
        assert hash(a) == hash(b)
        assert a != c

    def test_inf_nan_literals_hashable(self) -> None:
        assert hash(Literal(math.inf, "float")) == hash(Literal(math.inf, "float"))
        assert hash(Literal(math.nan, "float")) == hash(Literal(math.nan, "float"))

    def test_nested_nodes_hashable(self) -> None:
        p = Program(
            ("x",),
            (Binding("v", Call("sqrt", (Ref("x"),))),),
            Where(Compare("lt", Ref("x"), Literal(0, "int")), Literal(0.0, "float"), Ref("v")),
        )
        assert hash(p) == hash(
            Program(
                ("x",),
                (Binding("v", Call("sqrt", (Ref("x"),))),),
                Where(Compare("lt", Ref("x"), Literal(0, "int")), Literal(0.0, "float"), Ref("v")),
            )
        )

    def test_bool_is_not_int_literal(self) -> None:
        assert Literal(True, "bool") != Literal(1, "int")


class TestIsBool:
    def test_bool_nodes(self) -> None:
        assert is_bool(Compare("lt", Ref("x"), Literal(0, "int")))
        assert is_bool(UnaryOp("not", Ref("x")))
        assert is_bool(Logical("and", (Ref("a"), Ref("b"))))
        assert is_bool(Literal(True, "bool"))
        assert is_bool(Call("isnan", (Ref("x"),)))
        assert is_bool(Call("isinf", (Ref("x"),)))
        assert is_bool(Call("isfinite", (Ref("x"),)))

    def test_non_bool_nodes(self) -> None:
        assert not is_bool(Ref("x"))
        assert not is_bool(Literal(1, "int"))
        assert not is_bool(Literal(1.0, "float"))
        assert not is_bool(BinOp("add", Ref("x"), Ref("y")))
        assert not is_bool(UnaryOp("neg", Ref("x")))
        assert not is_bool(Call("sqrt", (Ref("x"),)))
        assert not is_bool(Where(Ref("c"), Ref("a"), Ref("b")))
        assert not is_bool(DType("int64"))


class TestSSAEnv:
    def test_first_binding_keeps_name(self) -> None:
        env = SSAEnv(["x"])
        assert env.bind("y") == "y"

    def test_rebind_picks_smallest_free_suffix(self) -> None:
        env = SSAEnv(["x"])
        env.bind("x")
        assert env.bind("x") == "x_1"
        assert env.bind("x") == "x_2"

    def test_rebind_skips_user_names(self) -> None:
        # user has variables x and x_1: rebinding x must skip x_1
        env = SSAEnv(["x", "x_1"])
        env.bind("x")
        assert env.bind("x") == "x_2"

    def test_reserved_namespace_name_mangled(self) -> None:
        # 'xp' is no longer reserved (the namespace var is renamed instead);
        # names the generated code cannot rename are still mangled
        env = SSAEnv(["xp"])
        assert env.bind("xp") == "xp"

    def test_reserved_hasattr_mangled(self) -> None:
        env = SSAEnv(["array_namespace"])
        assert env.bind("array_namespace") == "array_namespace_"
        env = SSAEnv(["hasattr"])
        assert env.bind("hasattr") == "hasattr_"

    def test_mangle_skips_taken_mangled_name(self) -> None:
        # user has both array_namespace and array_namespace_ variables
        env = SSAEnv(["array_namespace", "array_namespace_"])
        assert env.bind("array_namespace") == "array_namespace_1"

    def test_fresh_temp_skips_user_names(self) -> None:
        env = SSAEnv(["t_1"])
        assert env.fresh_temp("t") == "t_2"

    def test_fresh_temp_skips_emitted(self) -> None:
        env = SSAEnv([])
        env.fresh_temp("t")
        assert env.fresh_temp("t") == "t_2"

    def test_reserve_blocks_allocation(self) -> None:
        env = SSAEnv(["g"])
        env.reserve("g_vec")
        env.bind("g")
        # g_vec is reserved and must never be handed out as a binding name
        for _ in range(3):
            name = env.bind("g")
            assert "g_vec" not in name
