# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Codegen tests: IR -> source text via ast.unparse."""

from __future__ import annotations

import ast

from array_vectorize import codegen
from array_vectorize import ir
from array_vectorize.frontend import info
from array_vectorize.lower import types


def make_lowered(
    params: list[info.Param], program: ir.Program, name: str = "f", docstring: str | None = None
) -> types.LoweredFunction:
    return types.LoweredFunction(
        program=program,
        name=name,
        params=params,
        param_names=[p.name for p in params],
        hidden_params=[],
        helpers=[],
        namespace_var="xp",
        namespace_param="_namespace",
        emitted_names=frozenset(["x"]),
        source="def f(x):\n    return x",
        docstring=docstring,
    )


def gen(program: ir.Program, params: list[info.Param] | None = None) -> str:
    params = params if params is not None else [info.Param("x", "arg")]
    return codegen.generate_source(make_lowered(params, program), program)


def test_simple_return() -> None:
    src = gen(ir.Program(("x",), (), ir.BinOp("add", ir.Ref("x"), ir.Literal(1, "int"))))
    assert "return x + 1" in src


def test_module_shape() -> None:
    src = gen(ir.Program(("x",), (), ir.Ref("x")))
    assert src.startswith("from array_api_compat import array_namespace")
    assert "def f_vec(x):" in src
    assert '"""\n    (array-vectorized) no docstring on the scalar original.' in src
    assert "Notes:\n        Vectorized by array-vectorize from this scalar original::" in src
    assert "def f(x):\n                return x\n" in src
    assert "xp = array_namespace(" in src
    assert "hasattr" not in src


def test_module_shape_with_docstring() -> None:
    program = ir.Program(("x",), (), ir.Ref("x"))
    lowered = make_lowered([info.Param("x", "arg")], program, docstring="Double x.")
    src = codegen.generate_source(lowered, program)
    assert "(array-vectorized) Double x." in src
    assert "no docstring on the scalar original" not in src


def test_operators_stay_operators() -> None:
    src = gen(
        ir.Program(
            ("x", "y"),
            (),
            ir.BinOp(
                "sub",
                ir.BinOp("mul", ir.Ref("x"), ir.Ref("y")),
                ir.BinOp("div", ir.Ref("x"), ir.Ref("y")),
            ),
        ),
        [info.Param("x", "arg"), info.Param("y", "arg")],
    )
    assert "return x * y - x / y" in src


def test_div_pow_mod_floordiv_become_xp_calls() -> None:
    src = gen(ir.Program(("x",), (), ir.BinOp("pow", ir.Ref("x"), ir.Literal(2, "int"))))
    assert "xp.pow(x, 2)" in src
    src = gen(ir.Program(("x",), (), ir.BinOp("floordiv", ir.Ref("x"), ir.Literal(2, "int"))))
    assert "xp.floor_divide(x, 2)" in src
    src = gen(ir.Program(("x",), (), ir.BinOp("mod", ir.Ref("x"), ir.Literal(2, "int"))))
    assert "xp.remainder(x, 2)" in src


def test_logical_not_where() -> None:
    src = gen(
        ir.Program(
            ("x",),
            (),
            ir.Where(
                ir.UnaryOp("not", ir.Compare("gt", ir.Ref("x"), ir.Literal(0, "int"))),
                ir.Literal(1.0, "float"),
                ir.Ref("x"),
            ),
        )
    )
    assert "xp.where(xp.logical_not(x > 0), 1.0, x)" in src


def test_logical_and_folded_pairwise() -> None:
    src = gen(
        ir.Program(
            ("x",),
            (),
            ir.Logical(
                "and",
                (
                    ir.Compare("lt", ir.Literal(0, "int"), ir.Ref("x")),
                    ir.Compare("lt", ir.Ref("x"), ir.Literal(1, "int")),
                    ir.Compare("lt", ir.Ref("x"), ir.Literal(2, "int")),
                ),
            ),
        )
    )
    assert "xp.logical_and(xp.logical_and(0 < x, x < 1), x < 2)" in src


def test_inf_nan_literals() -> None:
    src = gen(
        ir.Program(("x",), (), ir.BinOp("add", ir.Ref("x"), ir.Literal(float("inf"), "float")))
    )
    assert "x + xp.inf" in src
    src = gen(
        ir.Program(("x",), (), ir.BinOp("add", ir.Ref("x"), ir.Literal(float("-inf"), "float")))
    )
    assert "x + -xp.inf" in src
    src = gen(
        ir.Program(("x",), (), ir.BinOp("add", ir.Ref("x"), ir.Literal(float("nan"), "float")))
    )
    assert "x + xp.nan" in src


def test_casts() -> None:
    src = gen(ir.Program(("x",), (), ir.Call("astype", (ir.Ref("x"), ir.DType("int64")))))
    assert "xp.astype(x, xp.int64)" in src


def test_bindings_and_temp() -> None:
    program = ir.Program(
        ("x",),
        (
            ir.Binding("t_1", ir.Call("sqrt", (ir.Ref("x"),))),
            ir.Binding("a", ir.BinOp("add", ir.Ref("t_1"), ir.Ref("t_1"))),
        ),
        ir.Ref("a"),
    )
    src = gen(program)
    assert "t_1 = xp.sqrt(x)" in src
    assert "a = t_1 + t_1" in src
    assert "return a" in src
    # statements in order
    assert src.index("t_1 =") < src.index("a =") < src.index("return a")


def test_defaults_and_kwonly() -> None:
    params = [
        info.Param("x", "arg"),
        info.Param("scale", "arg", 2.0, True),
        info.Param("flag", "kwonly", True, True),
    ]
    program = ir.Program(("x", "scale", "flag"), (), ir.Ref("x"))
    src = gen(program, params)
    assert "def f_vec(x, scale=2.0, *, flag=True):" in src


def test_hidden_params_kwonly_none() -> None:
    lowered = types.LoweredFunction(
        program=ir.Program(("x", "ARR"), (), ir.Ref("ARR")),
        name="f",
        params=[info.Param("x", "arg")],
        param_names=["x"],
        hidden_params=[("ARR", "placeholder")],
        helpers=[],
        namespace_var="xp",
        namespace_param="_namespace",
        emitted_names=frozenset(["x", "ARR"]),
        source="def f(x):\n    return ARR",
        docstring=None,
    )
    src = codegen.generate_source(lowered, lowered.program)
    assert "def f_vec(x, *, ARR=None):" in src
    assert "xp = array_namespace(x, ARR)" in src


def test_posonly_marker() -> None:
    params = [info.Param("x", "posonly"), info.Param("y", "arg")]
    program = ir.Program(("x", "y"), (), ir.Ref("x"))
    src = gen(program, params)
    assert "def f_vec(x, /, y):" in src


def test_reserved_param_kept_namespace_renamed() -> None:
    params = [info.Param("xp", "arg")]
    lowered = types.LoweredFunction(
        program=ir.Program(("xp",), (), ir.Ref("xp")),
        name="f",
        params=params,
        param_names=["xp"],
        hidden_params=[],
        helpers=[],
        namespace_var="xp_1",
        namespace_param="_namespace",
        emitted_names=frozenset(["xp"]),
        source="def f(xp):\n    return xp",
        docstring=None,
    )
    src = codegen.generate_source(lowered, lowered.program)
    assert "def f_vec(xp):" in src
    assert "xp_1 = array_namespace(" in src


def test_generated_source_is_valid_python() -> None:
    program = ir.Program(
        ("x", "y"),
        (
            ir.Binding("a", ir.Call("sqrt", (ir.BinOp("add", ir.Ref("x"), ir.Ref("y")),))),
            ir.Binding(
                "b",
                ir.Where(
                    ir.Compare("gt", ir.Ref("a"), ir.Literal(0, "int")),
                    ir.Ref("a"),
                    ir.Literal(0.0, "float"),
                ),
            ),
        ),
        ir.BinOp("add", ir.Ref("b"), ir.Literal(1, "int")),
    )
    src = gen(program, [info.Param("x", "arg"), info.Param("y", "arg")])
    ast.parse(src)  # must not raise


def test_no_numpy_leakage() -> None:
    src = gen(ir.Program(("x",), (), ir.Call("sqrt", (ir.Ref("x"),))))
    assert "np." not in src
    assert "numpy" not in src
