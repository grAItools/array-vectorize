"""Compat shim: canonical home is vectorizer.frontend.validate (restructure phase 2)."""

from .frontend.validate import (
    _ALLOWED_BINOPS,
    _ALLOWED_CMPOPS,
    _ALLOWED_UNARY,
    _EXPR_MESSAGES,
    _MATH_ATTRS,
    _STMT_MESSAGES,
    _TEMPLATE_STR,
    _Validator,
    validate,
)

__all__ = [
    "_ALLOWED_BINOPS",
    "_ALLOWED_CMPOPS",
    "_ALLOWED_UNARY",
    "_EXPR_MESSAGES",
    "_MATH_ATTRS",
    "_STMT_MESSAGES",
    "_TEMPLATE_STR",
    "_Validator",
    "validate",
]
