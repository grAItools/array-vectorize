"""Compat shim: canonical home is vectorizer.codegen (restructure phase 2)."""

from .codegen import generate_source
from .codegen.emitter import (
    _BINOP_AST,
    _BINOP_XP,
    _CMPOP_AST,
    _build_signature,
    _gen_expr,
    _gen_literal,
    _gen_range_call,
    _gen_stmt,
    _load,
    _namespace_line,
    _xp_attr,
    _xp_call,
)

__all__ = [
    "_BINOP_AST",
    "_BINOP_XP",
    "_CMPOP_AST",
    "_build_signature",
    "_gen_expr",
    "_gen_literal",
    "_gen_range_call",
    "_gen_stmt",
    "_load",
    "_namespace_line",
    "_xp_attr",
    "_xp_call",
    "generate_source",
]
