"""Compat shim: canonical home is vectorizer.frontend (restructure phase 2)."""

from ._errors import _reject
from .frontend.extract import (
    _MATH_FUNC_NAMES,
    FunctionInfo,
    Param,
    ParamKind,
    _check_default,
    _classify_closures,
    _params_of,
    extract_function,
)
from .frontend.info import _AstFunction
from .frontend.lambda_id import (
    _code_key,
    _const_key,
    _defaults_match,
    _find_target,
    _iter_code_objects,
    _lambdas_in_source_order,
)

__all__ = [
    "_MATH_FUNC_NAMES",
    "FunctionInfo",
    "Param",
    "ParamKind",
    "_AstFunction",
    "_check_default",
    "_classify_closures",
    "_code_key",
    "_const_key",
    "_defaults_match",
    "_find_target",
    "_iter_code_objects",
    "_lambdas_in_source_order",
    "_params_of",
    "_reject",
    "extract_function",
]
