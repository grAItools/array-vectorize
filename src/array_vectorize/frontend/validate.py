"""Supported-subset checking with linter-style diagnostics (design D5).

Structural checks only (AST shape); semantic checks (name resolution, call
arity, closure values) happen in extraction and lowering. All violations are
collected before raising.
"""

from __future__ import annotations

import ast

from array_vectorize.errors import Diagnostic, VectorizationError
from array_vectorize.frontend.info import FunctionInfo
from array_vectorize.frontend.tables import MATH_CONSTS, MATH_FUNCS, MATH_SPECIAL

__all__ = ["validate"]

_ALLOWED_BINOPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.FloorDiv,
    ast.Mod,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.LShift,
    ast.RShift,
)
_ALLOWED_UNARY = (ast.USub, ast.UAdd, ast.Invert, ast.Not)
_ALLOWED_CMPOPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)

_MATH_ATTRS = set(MATH_FUNCS) | set(MATH_SPECIAL) | set(MATH_CONSTS)

_STMT_MESSAGES: dict[type[ast.stmt], str] = {
    ast.While: "while loops are not supported (data-dependent control flow)",
    ast.Break: "break is not supported",
    ast.Continue: "continue is not supported",
    ast.Global: "global declarations are not supported",
    ast.Nonlocal: "nonlocal declarations are not supported",
    ast.ClassDef: "class definitions are not supported",
    ast.FunctionDef: "nested function definitions are not supported",
    ast.AsyncFunctionDef: "async functions are not supported",
    ast.Try: "try/except is not supported",
    ast.TryStar: "except* (exception groups) is not supported",
    ast.TypeAlias: "type alias statements (type X = ...) are not supported",
    ast.Raise: "raise is not supported",
    ast.Assert: "assert is not supported",
    ast.Import: "imports inside the function are not supported (import at module level)",
    ast.ImportFrom: "imports inside the function are not supported (import at module level)",
    ast.With: "with statements are not supported",
    ast.AsyncWith: "async with statements are not supported",
    ast.AsyncFor: "async for loops are not supported",
    ast.Match: "match statements are not supported",
    ast.Delete: "del statements are not supported",
    ast.AnnAssign: "annotated assignments are not supported",
}

#: t-strings (PEP 750) only exist in the grammar on Python 3.14+; the AST
#: node type itself varies by interpreter, so this is the one legitimate
#: version branch (the message table is built conditionally).
_TEMPLATE_STR = getattr(ast, "TemplateStr", None)

_EXPR_MESSAGES: dict[type[ast.expr], str] = {
    ast.Subscript: "subscripts are not supported",
    ast.Tuple: "tuples are not supported",
    ast.List: "lists are not supported",
    ast.Set: "sets are not supported",
    ast.Dict: "dicts are not supported",
    ast.ListComp: "list comprehensions are not supported",
    ast.SetComp: "set comprehensions are not supported",
    ast.DictComp: "dict comprehensions are not supported",
    ast.GeneratorExp: "generator expressions are not supported",
    ast.Lambda: "lambdas inside the function body are not supported",
    ast.NamedExpr: "walrus assignments are not supported",
    ast.JoinedStr: "f-strings are not supported",
    ast.Starred: "starred expressions are not supported",
    ast.Await: "await is not supported",
    ast.Slice: "slices are not supported",
    ast.Yield: "generators are not supported",
    ast.YieldFrom: "generators are not supported",
}
if _TEMPLATE_STR is not None:
    _EXPR_MESSAGES[_TEMPLATE_STR] = "template strings (t-strings) are not supported"


class _Validator:
    def __init__(self, info: FunctionInfo) -> None:
        self.info = info
        self.lines = info.source.splitlines()
        self.diags: list[Diagnostic] = []
        self.saw_return = False

    def error(self, node: ast.AST, message: str) -> None:
        lineno = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        line = self.lines[lineno - 1] if 0 < lineno <= len(self.lines) else ""
        self.diags.append(Diagnostic(message, lineno, col, line))

    # ------------------------------------------------------------- statements

    def validate_stmts(self, stmts: list[ast.stmt]) -> None:
        seen_return = False
        for i, stmt in enumerate(stmts):
            if seen_return:
                self.error(stmt, "statements after return are not supported")
            self.validate_stmt(stmt, docstring_ok=(i == 0))
            if isinstance(stmt, ast.Return):
                seen_return = True
                self.saw_return = True

    def validate_stmt(self, stmt: ast.stmt, *, docstring_ok: bool) -> None:
        match stmt:
            case ast.Assign():
                self.validate_assign_target(stmt)
                self.validate_expr(stmt.value)
            case ast.AugAssign():
                self.validate_assign_target(stmt)
                if not isinstance(stmt.op, _ALLOWED_BINOPS):
                    self.error(
                        stmt, f"augmented assignment {type(stmt.op).__name__} is not supported"
                    )
                self.validate_expr(stmt.value)
            case ast.Return():
                if stmt.value is None:
                    self.error(stmt, "bare returns are not supported; return a value")
                else:
                    self.validate_expr(stmt.value)
            case ast.Expr():
                if (
                    docstring_ok
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                ):
                    return  # leading docstring
                self.error(
                    stmt, "expression statements are not supported (only the leading docstring)"
                )
                if not isinstance(stmt.value, ast.Constant):
                    self.validate_expr(stmt.value)
            case ast.Pass():
                return
            case ast.If():
                self.validate_expr(stmt.test)
                self.validate_stmts(stmt.body)
                self.validate_stmts(stmt.orelse)
            case ast.For():
                self.validate_for(stmt)
            case _:
                # ast.stmt is an open set (new syntax per Python release):
                # reject unknown statements with the message table
                msg = _STMT_MESSAGES.get(type(stmt))
                if msg is None:
                    msg = f"{type(stmt).__name__} statements are not supported"
                self.error(stmt, msg)

    def validate_assign_target(self, stmt: ast.Assign | ast.AugAssign) -> None:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1:
                self.error(stmt, "multiple assignment targets are not supported")
                return
            target = stmt.targets[0]
        else:
            target = stmt.target
        if isinstance(target, ast.Name):
            return
        if isinstance(target, ast.Tuple | ast.List):
            self.error(stmt, "tuple/list assignment is not supported (single name targets only)")
        elif isinstance(target, ast.Starred):  # pragma: no cover - syntax error in Python
            self.error(stmt, "starred assignment is not supported")
        elif isinstance(target, ast.Subscript):
            self.error(stmt, "subscript assignment is not supported")
        elif isinstance(target, ast.Attribute):
            self.error(stmt, "attribute assignment is not supported")
        else:
            self.error(stmt, "this assignment target is not supported")

    def validate_for(self, stmt: ast.For) -> None:
        if not isinstance(stmt.target, ast.Name):
            self.error(stmt, "for loop targets must be a single name")
        if stmt.orelse:
            self.error(stmt, "for/else is not supported")
        iter_ = stmt.iter
        if (
            isinstance(iter_, ast.Call)
            and isinstance(iter_.func, ast.Name)
            and iter_.func.id == "range"
            and not iter_.keywords
            and 1 <= len(iter_.args) <= 3
        ):
            # bound constancy is a semantic property, checked in lowering
            for a in iter_.args:
                self.validate_expr(a)
        else:
            self.error(stmt, "only 'for i in range(...)' with literal/name bounds is supported")
        self.validate_stmts(stmt.body)

    # ------------------------------------------------------------ expressions

    def validate_expr(self, node: ast.expr) -> None:
        match node:
            case ast.Constant():
                if type(node.value) not in (int, float, bool):
                    self.error(node, f"{type(node.value).__name__!r} literals are not supported")
                elif type(node.value) is int and not (-(2**63) <= node.value < 2**63):
                    self.error(node, "int literals outside the int64 range are not supported")
            case ast.Name():
                return
            case ast.BinOp():
                if not isinstance(node.op, _ALLOWED_BINOPS):
                    self.error(node, f"operator {type(node.op).__name__} is not supported")
                self.validate_expr(node.left)
                self.validate_expr(node.right)
            case ast.UnaryOp():
                if not isinstance(node.op, _ALLOWED_UNARY):
                    self.error(node, f"unary operator {type(node.op).__name__} is not supported")
                self.validate_expr(node.operand)
            case ast.BoolOp():
                for v in node.values:
                    self.validate_expr(v)
            case ast.Compare():
                if len(node.ops) > 1:
                    for op in node.ops:
                        if not isinstance(op, _ALLOWED_CMPOPS):
                            self.error(
                                node, f"comparison {type(op).__name__} in chain is not supported"
                            )
                elif not isinstance(node.ops[0], _ALLOWED_CMPOPS):
                    self.error(
                        node,
                        f"comparison {type(node.ops[0]).__name__} is not supported "
                        "(only == != < <= > >=)",
                    )
                self.validate_expr(node.left)
                for comp in node.comparators:
                    self.validate_expr(comp)
            case ast.IfExp():
                self.validate_expr(node.test)
                self.validate_expr(node.body)
                self.validate_expr(node.orelse)
            case ast.Call():
                self.validate_call(node)
            case ast.Attribute():
                self.validate_attribute(node)
            case _:
                # ast.expr is an open set (new syntax per Python release):
                # reject unknown expressions with the message table
                msg = _EXPR_MESSAGES.get(type(node))
                if msg is None:
                    msg = f"{type(node).__name__} expressions are not supported"
                self.error(node, msg)

    def validate_attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in self.info.math_modules:
            if node.attr in _MATH_ATTRS:
                return
            self.error(node, f"math.{node.attr} is not in the supported math subset")
            return
        self.error(node, "attribute access is not supported (only math.<fn> for mapped functions)")

    def validate_call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            self.validate_attribute(node.func)
        elif isinstance(node.func, ast.Name):
            pass  # resolution is semantic; checked in lowering
        else:
            self.error(node, "calling this expression is not supported")
            self.validate_expr(node.func)
        if node.keywords:
            self.error(node, "keyword arguments in calls are not supported (positional only)")
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                self.error(node, "starred call arguments are not supported")
            else:
                self.validate_expr(arg)


def validate(info: FunctionInfo) -> None:
    """Validate the supported subset; raise with all violations if any."""
    v = _Validator(info)
    tree = info.tree
    if isinstance(tree, ast.FunctionDef):
        if tree.type_params:
            # PEP 695: `def f[T](x)`; nested generic defs are already rejected
            # as nested function definitions
            v.error(tree, "generic functions (type parameters) are not supported")
        v.validate_stmts(tree.body)
        if not v.saw_return:
            v.error(tree, "function has no return statement; it must return a value")
    else:  # ast.Lambda
        v.validate_expr(tree.body)
    if v.diags:
        n = len(v.diags)
        raise VectorizationError(
            f"cannot vectorize {info.name!r}: {n} unsupported construct"
            f"{'' if n == 1 else 's'} found",
            v.diags,
        )
