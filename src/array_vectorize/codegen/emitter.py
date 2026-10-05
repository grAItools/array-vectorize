"""IR -> readable Python source (plan §9).

Builds an ``ast.Module`` and uses ``ast.unparse``: correct docstring/literal
escaping for free, normalized stable formatting. Generated code contains only
``xp.*`` calls plus one ``from array_api_compat import array_namespace`` —
except in pinned mode (``vectorize(namespace=...)``), where the namespace is
taken from a hidden keyword-only parameter instead and the import is omitted
(pinned generated source has zero imports).
"""

from __future__ import annotations

import ast
import math
from typing import assert_never

from ..ir import (
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
from ..lower.types import LoweredFunction

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


def _xp_attr(ns: str, attr: str) -> ast.Attribute:
    return ast.Attribute(value=_load(ns), attr=attr, ctx=ast.Load())


def _xp_call(ns: str, attr: str, args: list[ast.expr]) -> ast.Call:
    return ast.Call(func=_xp_attr(ns, attr), args=args, keywords=[])


def _gen_literal(lit: Literal, ns: str) -> ast.expr:
    value = lit.value
    if isinstance(value, float):
        if math.isnan(value):
            return _xp_attr(ns, "nan")
        if math.isinf(value):
            if value > 0:
                return _xp_attr(ns, "inf")
            return ast.UnaryOp(op=ast.USub(), operand=_xp_attr(ns, "inf"))
    return ast.Constant(value=value)


def _gen_expr(node: Node, ns: str) -> ast.expr:
    match node:
        case Literal():
            return _gen_literal(node, ns)
        case Ref():
            return _load(node.name)
        case DType():
            return _xp_attr(ns, node.name)
        case BinOp():
            if node.op in _BINOP_AST:
                return ast.BinOp(
                    op=_BINOP_AST[node.op](),
                    left=_gen_expr(node.left, ns),
                    right=_gen_expr(node.right, ns),
                )
            return _xp_call(
                ns, _BINOP_XP[node.op], [_gen_expr(node.left, ns), _gen_expr(node.right, ns)]
            )
        case UnaryOp():
            if node.op == "neg":
                return ast.UnaryOp(op=ast.USub(), operand=_gen_expr(node.operand, ns))
            if node.op == "pos":
                return ast.UnaryOp(op=ast.UAdd(), operand=_gen_expr(node.operand, ns))
            if node.op == "invert":
                return ast.UnaryOp(op=ast.Invert(), operand=_gen_expr(node.operand, ns))
            return _xp_call(ns, "logical_not", [_gen_expr(node.operand, ns)])
        case Compare():
            return ast.Compare(
                left=_gen_expr(node.left, ns),
                ops=[_CMPOP_AST[node.op]()],
                comparators=[_gen_expr(node.right, ns)],
            )
        case Logical():
            fn = "logical_and" if node.op == "and" else "logical_or"
            folded = _gen_expr(node.parts[0], ns)
            for part in node.parts[1:]:
                folded = _xp_call(ns, fn, [folded, _gen_expr(part, ns)])
            return folded
        case Where():
            return _xp_call(
                ns,
                "where",
                [_gen_expr(node.cond, ns), _gen_expr(node.then, ns), _gen_expr(node.other, ns)],
            )
        case Call():
            return _xp_call(ns, node.fn, [_gen_expr(a, ns) for a in node.args])
        case FuncCall():
            return ast.Call(
                func=_load(node.fn), args=[_gen_expr(a, ns) for a in node.args], keywords=[]
            )
        case _:
            assert_never(node)


def _namespace_line(ns: str, param_names: list[str]) -> ast.Assign:
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
    return ast.Assign(targets=[ast.Name(id=ns, ctx=ast.Store())], value=call)


def _build_signature(lowered: LoweredFunction, namespace_param: str | None = None) -> ast.arguments:
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
    if namespace_param is not None:
        # pinned mode: the hidden namespace parameter, mirroring how
        # hidden_params are emitted (the real default is injected at runtime)
        kwonly.append(ast.arg(arg=namespace_param, annotation=None))
        kw_defaults.append(ast.Constant(value=None))
    return ast.arguments(
        posonlyargs=posonly,
        args=positional,
        vararg=None,
        kwonlyargs=kwonly,
        kw_defaults=kw_defaults,
        kwarg=None,
        defaults=defaults,
    )


def _gen_range_call(loop: Loop, ns: str) -> ast.Call:
    step_one = isinstance(loop.step, Literal) and loop.step.value == 1
    args = [_gen_expr(loop.start, ns), _gen_expr(loop.stop, ns)]
    if not step_one:
        args.append(_gen_expr(loop.step, ns))
    return ast.Call(func=_load("range"), args=args, keywords=[])


def _gen_stmt(stmt: Stmt, ns: str) -> ast.stmt:
    match stmt:
        case Binding():
            return ast.Assign(
                targets=[ast.Name(id=stmt.name, ctx=ast.Store())],
                value=_gen_expr(stmt.expr, ns),
            )
        case Loop():
            body = [_gen_stmt(inner, ns) for inner in stmt.body]
            if not body:
                # (15) a retained loop with a fully dead body still needs a statement
                body = [ast.Pass()]
            return ast.For(
                target=ast.Name(id=stmt.var, ctx=ast.Store()),
                iter=_gen_range_call(stmt, ns),
                body=body,
                orelse=[],
            )
        case _:
            assert_never(stmt)


def generate_source(lowered: LoweredFunction, program: Program, *, pinned: bool = False) -> str:
    """Generate the vectorized function source (plan §9).

    With ``pinned=True`` (``vectorize(namespace=...)``) the namespace is
    never extracted from the arguments: ``xp`` binds directly to the hidden
    keyword-only ``namespace_param``, whose real default is injected at
    runtime, and the ``array_namespace`` import is omitted.
    """
    func_name = generated_name(lowered.name)
    ns = lowered.namespace_var
    all_params = lowered.param_names + [name for name, _ in lowered.hidden_params]

    if pinned:
        # xp = _namespace (the hidden kw-only parameter's runtime default)
        ns_line: ast.stmt = ast.Assign(
            targets=[ast.Name(id=ns, ctx=ast.Store())], value=_load(lowered.namespace_param)
        )
    else:
        ns_line = _namespace_line(ns, all_params)
    body: list[ast.stmt] = [
        ast.Expr(value=ast.Constant(value=lowered.source)),  # original source, verbatim
        ns_line,
    ]
    for stmt in program.bindings:
        body.append(_gen_stmt(stmt, ns))
    body.append(ast.Return(value=_gen_expr(program.result, ns)))

    module_body: list[ast.stmt] = []
    if not pinned:
        module_body.append(
            ast.ImportFrom(
                module="array_api_compat",
                names=[ast.alias(name="array_namespace", asname=None)],
                level=0,
            )
        )
    module_body.append(
        ast.FunctionDef(
            name=func_name,
            args=_build_signature(
                lowered, namespace_param=lowered.namespace_param if pinned else None
            ),
            body=body,
            decorator_list=[],
            returns=None,
            type_comment=None,
            type_params=[],
        )
    )
    module = ast.Module(body=module_body, type_ignores=[])
    ast.fix_missing_locations(module)
    return ast.unparse(module) + "\n"
