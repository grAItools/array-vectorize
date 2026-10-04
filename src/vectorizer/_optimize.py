"""Compat shim: canonical home is vectorizer.optimize (restructure phase 2)."""

from .optimize import const_fold, cse, dce, optimize, protect_domains
from .optimize.constfold import (
    _CMPOP_PY,
    _FOLDABLE_CALLS,
    _NO_FOLD_OPS,
    _ZERO_RISK_OPS,
    _const_fold_expr,
    _fold_binop,
    _fold_compare,
    _fold_stmt,
    _fold_unary,
    _int64_ok,
    _literal_kind,
)
from .optimize.cse import _CSE_ELIGIBLE, _count_eligible, _has_loop
from .optimize.dce import _mark_live, _mark_live_stmt
from .optimize.protect_domains import (
    _PARTIAL_DOMAINS,
    _and_ctx,
    _clamp_dead,
    _free_names,
)

__all__ = [
    "_CMPOP_PY",
    "_CSE_ELIGIBLE",
    "_FOLDABLE_CALLS",
    "_NO_FOLD_OPS",
    "_PARTIAL_DOMAINS",
    "_ZERO_RISK_OPS",
    "_and_ctx",
    "_clamp_dead",
    "_const_fold_expr",
    "_count_eligible",
    "_fold_binop",
    "_fold_compare",
    "_fold_stmt",
    "_fold_unary",
    "_free_names",
    "_has_loop",
    "_int64_ok",
    "_literal_kind",
    "_mark_live",
    "_mark_live_stmt",
    "const_fold",
    "cse",
    "dce",
    "optimize",
    "protect_domains",
]
