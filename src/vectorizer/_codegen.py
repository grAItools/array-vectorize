"""IR -> readable Python source (plan §9).

Builds an ``ast.Module`` and uses ``ast.unparse``: correct docstring/literal
escaping for free, normalized stable formatting. Generated code contains only
``xp.*`` calls plus one ``from array_api_compat import array_namespace``.
"""

from __future__ import annotations

import ast
import math

from ._ir import (
    Binding,
    BinOp,
    Call,
    Compare,
    DType,
    FuncCall,
    Literal,
    Logical,
    Loop,
    Node,
    Program,
    Ref,
    Stmt,
    UnaryOp,
    Where,
    generated_name,  # re-export helper
)
from ._lower import LoweredFunction

__all__ = ["generate_source"]


_BINOP_AST: dict[str, type[ast.operator]] = {
    "add": ast.Add,
    "sub": ast.Sub,
    "mul": ast.Mult,
    "div": ast.Div,
    "and": ast.BitAnd,
    "or": ast.BitOr,
    "xor": ast.BitXor,
    "lshift": ast.LShift,
    "rshift": ast.RShift,
}
_BINOP_XP: dict[str, str] = {
    "pow": "pow",
    "floordiv": "floor_divide",
    "mod": "remainder",
}
_CMPOP_AST: dict[str, type[ast.cmpop]] = {
    "eq": ast.Eq,
    "ne": ast.NotEq,
    "lt": ast.Lt,
    "le": ast.LtE,
    "gt": ast.Gt,
    "ge": ast.GtE,
}


def _load(name: str) -> ast.Name:
    return ast.Name(id=name, ctx=ast.Load())


def _xp_attr(attr: str) -> ast.Attribute:
    return ast.Attribute(value=_load("xp"), attr=attr, ctx=ast.Load())


def _xp_call(attr: str, args: list[ast.expr]) -> ast.Call:
    return ast.Call(func=_xp_attr(attr), args=args, keywords=[])


def _gen_literal(lit: Literal) -> ast.expr:
    value = lit.value
    if isinstance(value, float):
        if math.isnan(value):
            return _xp_attr("nan")
        if math.isinf(value):
            if value > 0:
                return _xp_attr("inf")
            return ast.UnaryOp(op=ast.USub(), operand=_xp_attr("inf"))
    return ast.Constant(value=value)


def _gen_expr(node: Node) -> ast.expr:
    if isinstance(node, Literal):
        return _gen_literal(node)
    if isinstance(node, Ref):
        return _load(node.name)
    if isinstance(node, DType):
        return _xp_attr(node.name)
    if isinstance(node, BinOp):
        if node.op in _BINOP_AST:
            return ast.BinOp(
                op=_BINOP_AST[node.op](),
                left=_gen_expr(node.left),
                right=_gen_expr(node.right),
            )
        return _xp_call(_BINOP_XP[node.op], [_gen_expr(node.left), _gen_expr(node.right)])
    if isinstance(node, UnaryOp):
        if node.op == "neg":
            return ast.UnaryOp(op=ast.USub(), operand=_gen_expr(node.operand))
        if node.op == "pos":
            return ast.UnaryOp(op=ast.UAdd(), operand=_gen_expr(node.operand))
        if node.op == "invert":
            return ast.UnaryOp(op=ast.Invert(), operand=_gen_expr(node.operand))
        return _xp_call("logical_not", [_gen_expr(node.operand)])
    if isinstance(node, Compare):
        return ast.Compare(
            left=_gen_expr(node.left),
            ops=[_CMPOP_AST[node.op]()],
            comparators=[_gen_expr(node.right)],
        )
    if isinstance(node, Logical):
        fn = "logical_and" if node.op == "and" else "logical_or"
        folded = _gen_expr(node.parts[0])
        for part in node.parts[1:]:
            folded = _xp_call(fn, [folded, _gen_expr(part)])
        return folded
    if isinstance(node, Where):
        return _xp_call(
            "where", [_gen_expr(node.cond), _gen_expr(node.then), _gen_expr(node.other)]
        )
    if isinstance(node, Call):
        return _xp_call(node.fn, [_gen_expr(a) for a in node.args])
    if isinstance(node, FuncCall):
        return ast.Call(func=_load(node.fn), args=[_gen_expr(a) for a in node.args], keywords=[])
    raise TypeError(f"unexpected IR node {type(node).__name__}")


def _namespace_line(param_names: list[str]) -> ast.Assign:
    # xp = array_namespace(*[a for a in (x, y) if hasattr(a, "__array_namespace__")])
    # 'a' is comprehension-scoped, so it cannot collide with user names.
    element = _load("a")
    generator = ast.comprehension(
        target=ast.Name(id="a", ctx=ast.Store()),
        iter=ast.Tuple(elts=[_load(p) for p in param_names], ctx=ast.Load()),
        ifs=[
            ast.Call(
                func=_load("hasattr"),
                args=[_load("a"), ast.Constant(value="__array_namespace__")],
                keywords=[],
            )
        ],
        is_async=False,
    )
    filtered = ast.ListComp(elt=element, generators=[generator])
    call = ast.Call(
        func=_load("array_namespace"),
        args=[ast.Starred(value=filtered, ctx=ast.Load())],
        keywords=[],
    )
    return ast.Assign(targets=[ast.Name(id="xp", ctx=ast.Store())], value=call)


def _build_signature(lowered: LoweredFunction) -> ast.arguments:
    posonly: list[ast.arg] = []
    positional: list[ast.arg] = []
    defaults: list[ast.expr] = []
    kwonly: list[ast.arg] = []
    kw_defaults: list[ast.expr | None] = []

    seen_default = False
    for param, emitted in zip(lowered.params, lowered.param_names, strict=True):
        arg = ast.arg(arg=emitted, annotation=None)
        if param.kind == "kwonly":
            kwonly.append(arg)
            kw_defaults.append(ast.Constant(value=param.default) if param.has_default else None)
            continue
        if param.kind == "posonly":
            posonly.append(arg)
        else:
            positional.append(arg)
        if param.has_default:
            defaults.append(ast.Constant(value=param.default))
            seen_default = True
        elif seen_default:
            # positional non-default after default cannot occur in valid Python
            raise ValueError(f"parameter {param.name!r} lacks a default after one")
    for name, _default in lowered.hidden_params:
        kwonly.append(ast.arg(arg=name, annotation=None))
        kw_defaults.append(ast.Constant(value=None))  # real default injected at runtime
    return ast.arguments(
        posonlyargs=posonly,
        args=positional,
        vararg=None,
        kwonlyargs=kwonly,
        kw_defaults=kw_defaults,
        kwarg=None,
        defaults=defaults,
    )


def _gen_range_call(loop: Loop) -> ast.Call:
    step_one = isinstance(loop.step, Literal) and loop.step.value == 1
    args = [_gen_expr(loop.start), _gen_expr(loop.stop)]
    if not step_one:
        args.append(_gen_expr(loop.step))
    return ast.Call(func=_load("range"), args=args, keywords=[])


def _gen_stmt(stmt: Stmt) -> ast.stmt:
    if isinstance(stmt, Binding):
        return ast.Assign(
            targets=[ast.Name(id=stmt.name, ctx=ast.Store())],
            value=_gen_expr(stmt.expr),
        )
    assert isinstance(stmt, Loop)
    return ast.For(
        target=ast.Name(id=stmt.var, ctx=ast.Store()),
        iter=_gen_range_call(stmt),
        body=[_gen_stmt(s) for s in stmt.body],
        orelse=[],
    )


def generate_source(lowered: LoweredFunction, program: Program) -> str:
    """Generate the vectorized function source (plan §9)."""
    func_name = generated_name(lowered.name)
    all_params = lowered.param_names + [name for name, _ in lowered.hidden_params]

    body: list[ast.stmt] = [
        ast.Expr(value=ast.Constant(value=lowered.source)),  # original source, verbatim
        _namespace_line(all_params),
    ]
    for stmt in program.bindings:
        body.append(_gen_stmt(stmt))
    body.append(ast.Return(value=_gen_expr(program.result)))

    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="array_api_compat",
                names=[ast.alias(name="array_namespace", asname=None)],
                level=0,
            ),
            ast.FunctionDef(
                name=func_name,
                args=_build_signature(lowered),
                body=body,
                decorator_list=[],
                returns=None,
                type_comment=None,
            ),
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    return ast.unparse(module) + "\n"
