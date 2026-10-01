"""Codegen tests: IR -> source text via ast.unparse (plan §9)."""

from __future__ import annotations

import ast

from vectorizer._codegen import generate_source
from vectorizer._extract import Param
from vectorizer._ir import (
    Binding,
    BinOp,
    Call,
    Compare,
    DType,
    Literal,
    Logical,
    Program,
    Ref,
    UnaryOp,
    Where,
)
from vectorizer._lower import LoweredFunction


def make_lowered(params: list[Param], program: Program, name: str = "f") -> LoweredFunction:
    return LoweredFunction(
        program=program,
        name=name,
        params=params,
        param_names=[p.name for p in params],
        hidden_params=[],
        source="def f(x):\n    return x",
    )


def gen(program: Program, params: list[Param] | None = None) -> str:
    params = params if params is not None else [Param("x", "arg")]
    return generate_source(make_lowered(params, program), program)


def test_simple_return() -> None:
    src = gen(Program(("x",), (), BinOp("add", Ref("x"), Literal(1, "int"))))
    assert "return x + 1" in src


def test_module_shape() -> None:
    src = gen(Program(("x",), (), Ref("x")))
    assert src.startswith("from array_api_compat import array_namespace")
    assert "def f_vec(x):" in src
    assert '"""def f(x):\n    return x"""' in src
    assert "xp = array_namespace(" in src
    assert "hasattr(a, '__array_namespace__')" in src


def test_operators_stay_operators() -> None:
    src = gen(
        Program(
            ("x", "y"),
            (),
            BinOp(
                "sub",
                BinOp("mul", Ref("x"), Ref("y")),
                BinOp("div", Ref("x"), Ref("y")),
            ),
        ),
        [Param("x", "arg"), Param("y", "arg")],
    )
    assert "return x * y - x / y" in src


def test_div_pow_mod_floordiv_become_xp_calls() -> None:
    src = gen(Program(("x",), (), BinOp("pow", Ref("x"), Literal(2, "int"))))
    assert "xp.pow(x, 2)" in src
    src = gen(Program(("x",), (), BinOp("floordiv", Ref("x"), Literal(2, "int"))))
    assert "xp.floor_divide(x, 2)" in src
    src = gen(Program(("x",), (), BinOp("mod", Ref("x"), Literal(2, "int"))))
    assert "xp.remainder(x, 2)" in src


def test_logical_not_where() -> None:
    src = gen(
        Program(
            ("x",),
            (),
            Where(
                UnaryOp("not", Compare("gt", Ref("x"), Literal(0, "int"))),
                Literal(1.0, "float"),
                Ref("x"),
            ),
        )
    )
    assert "xp.where(xp.logical_not(x > 0), 1.0, x)" in src


def test_logical_and_folded_pairwise() -> None:
    src = gen(
        Program(
            ("x",),
            (),
            Logical(
                "and",
                (
                    Compare("lt", Literal(0, "int"), Ref("x")),
                    Compare("lt", Ref("x"), Literal(1, "int")),
                    Compare("lt", Ref("x"), Literal(2, "int")),
                ),
            ),
        )
    )
    assert "xp.logical_and(xp.logical_and(0 < x, x < 1), x < 2)" in src


def test_inf_nan_literals() -> None:
    src = gen(Program(("x",), (), BinOp("add", Ref("x"), Literal(float("inf"), "float"))))
    assert "x + xp.inf" in src
    src = gen(Program(("x",), (), BinOp("add", Ref("x"), Literal(float("-inf"), "float"))))
    assert "x + -xp.inf" in src
    src = gen(Program(("x",), (), BinOp("add", Ref("x"), Literal(float("nan"), "float"))))
    assert "x + xp.nan" in src


def test_casts() -> None:
    src = gen(Program(("x",), (), Call("astype", (Ref("x"), DType("int64")))))
    assert "xp.astype(x, xp.int64)" in src


def test_bindings_and_temp() -> None:
    program = Program(
        ("x",),
        (
            Binding("t_1", Call("sqrt", (Ref("x"),))),
            Binding("a", BinOp("add", Ref("t_1"), Ref("t_1"))),
        ),
        Ref("a"),
    )
    src = gen(program)
    assert "t_1 = xp.sqrt(x)" in src
    assert "a = t_1 + t_1" in src
    assert "return a" in src
    # statements in order
    assert src.index("t_1 =") < src.index("a =") < src.index("return a")


def test_defaults_and_kwonly() -> None:
    params = [
        Param("x", "arg"),
        Param("scale", "arg", 2.0, True),
        Param("flag", "kwonly", True, True),
    ]
    program = Program(("x", "scale", "flag"), (), Ref("x"))
    src = gen(program, params)
    assert "def f_vec(x, scale=2.0, *, flag=True):" in src


def test_hidden_params_kwonly_none() -> None:
    lowered = LoweredFunction(
        program=Program(("x", "ARR"), (), Ref("ARR")),
        name="f",
        params=[Param("x", "arg")],
        param_names=["x"],
        hidden_params=[("ARR", "placeholder")],
        source="def f(x):\n    return ARR",
    )
    src = generate_source(lowered, lowered.program)
    assert "def f_vec(x, *, ARR=None):" in src
    assert "hasattr(a, '__array_namespace__')" in src


def test_posonly_marker() -> None:
    params = [Param("x", "posonly"), Param("y", "arg")]
    program = Program(("x", "y"), (), Ref("x"))
    src = gen(program, params)
    assert "def f_vec(x, /, y):" in src


def test_reserved_param_mangled_in_signature() -> None:
    params = [Param("xp", "arg")]
    lowered = LoweredFunction(
        program=Program(("xp_",), (), Ref("xp_")),
        name="f",
        params=params,
        param_names=["xp_"],
        hidden_params=[],
        source="def f(xp):\n    return xp",
    )
    src = generate_source(lowered, lowered.program)
    assert "def f_vec(xp_):" in src
    assert "return xp_" in src


def test_generated_source_is_valid_python() -> None:
    program = Program(
        ("x", "y"),
        (
            Binding("a", Call("sqrt", (BinOp("add", Ref("x"), Ref("y")),))),
            Binding(
                "b",
                Where(Compare("gt", Ref("a"), Literal(0, "int")), Ref("a"), Literal(0.0, "float")),
            ),
        ),
        BinOp("add", Ref("b"), Literal(1, "int")),
    )
    src = gen(program, [Param("x", "arg"), Param("y", "arg")])
    ast.parse(src)  # must not raise


def test_no_numpy_leakage() -> None:
    src = gen(Program(("x",), (), Call("sqrt", (Ref("x"),))))
    assert "np." not in src
    assert "numpy" not in src
