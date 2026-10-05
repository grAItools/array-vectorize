"""Source extraction and closure/constant capture (plan §4)."""

from __future__ import annotations

import ast
import inspect
import linecache
import math
import textwrap
import types
from collections.abc import Callable
from typing import Any

from ..compat import _is_array
from ..errors import _reject
from .info import FunctionInfo, Param, ParamKind, _AstFunction
from .lambda_id import _find_target
from .tables import MATH_FUNCS, MATH_SPECIAL

__all__ = ["FunctionInfo", "Param", "ParamKind", "extract_function"]

_MATH_FUNC_NAMES = (*MATH_FUNCS, *MATH_SPECIAL)


def _check_default(param: str, node: ast.expr) -> int | float | bool:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float, bool):
        return node.value  # type: ignore[return-value]
    _reject(param, f"default values must be int/float/bool literals (got {ast.unparse(node)!r})")


def _params_of(fn: _AstFunction, name: str) -> list[Param]:
    a = fn.args
    if a.vararg is not None or a.kwarg is not None:
        _reject(name, "*args/**kwargs are not supported")
    params: list[Param] = []
    positional = [*a.posonlyargs, *a.args]
    defaults: dict[int, ast.expr] = {}
    offset = len(positional) - len(a.defaults)
    for i, default in enumerate(a.defaults):
        defaults[offset + i] = default
    for i, arg in enumerate(positional):
        kind: ParamKind = "posonly" if i < len(a.posonlyargs) else "arg"
        if i in defaults:
            params.append(Param(arg.arg, kind, _check_default(arg.arg, defaults[i]), True))
        else:
            params.append(Param(arg.arg, kind, None, False))
    for arg, kw_default in zip(a.kwonlyargs, a.kw_defaults, strict=True):
        if kw_default is not None:
            params.append(Param(arg.arg, "kwonly", _check_default(arg.arg, kw_default), True))
        else:
            params.append(Param(arg.arg, "kwonly", None, False))
    if not params:
        _reject(name, "zero-argument functions cannot be vectorized")
    return params


def _classify_closures(target: types.FunctionType, info: FunctionInfo) -> None:
    # getclosurevars works off co_names, which includes attribute names
    # (``math.exp`` contributes ``exp``); filter to actual Name references.
    loaded = {n.id for n in ast.walk(info.tree) if isinstance(n, ast.Name)}
    cv = inspect.getclosurevars(target)
    resolved: dict[str, Any] = {k: v for k, v in cv.globals.items() if k in loaded}
    resolved.update({k: v for k, v in cv.nonlocals.items() if k in loaded})
    unbound = cv.unbound & loaded
    if unbound:
        names = ", ".join(sorted(unbound))
        _reject(info.name, f"name(s) not resolvable in the function's scope: {names}")
    for var, value in resolved.items():
        if value is math:
            info.math_modules.add(var)
        elif any(value is getattr(math, m) for m in _MATH_FUNC_NAMES):
            attr = next(m for m in _MATH_FUNC_NAMES if value is getattr(math, m))
            info.math_funcs[var] = attr
        elif isinstance(value, bool | int | float):
            info.closure_scalars[var] = value
        elif _is_array(value):
            info.closure_arrays[var] = value
        elif isinstance(value, types.FunctionType):
            info.user_funcs[var] = value
        else:
            _reject(
                info.name,
                f"closure/global {var!r} has unsupported type {type(value).__name__!r}; "
                "supported: int/float/bool scalars, Array API arrays, math functions, "
                "or plain Python functions",
            )


def extract_function(func: Callable[..., Any]) -> FunctionInfo:
    """Unwrap, read source, parse, and capture closures (plan §4)."""
    target = inspect.unwrap(func)
    # follow our own marker: vectorize(vectorize(f)) re-extracts the original
    while True:
        original = getattr(target, "_vectorized_original", None)
        if not isinstance(original, types.FunctionType):
            break
        target = original
    if not isinstance(target, types.FunctionType):
        if inspect.isbuiltin(target):
            _reject(
                getattr(target, "__name__", repr(target)),
                "C-implemented builtins have no inspectable Python source",
            )
        _reject(
            getattr(target, "__name__", repr(target)),
            f"expected a plain Python function, got {type(target).__name__}",
        )
    name = target.__name__
    try:
        raw = inspect.getsource(target)
    except (OSError, TypeError):
        _reject(
            name,
            "source is not available (defined in a REPL or via exec()?); "
            "define the function in a .py file so inspect.getsource works",
        )
    try:
        filename = inspect.getsourcefile(target) or inspect.getfile(target)
    except OSError:
        filename = "<unknown>"

    module_source = None
    if filename and filename != "<unknown>":
        module_source = "".join(linecache.getlines(filename))
    tree = _find_target(
        ast.parse(textwrap.dedent(raw)),
        name,
        target,
        module_source=module_source,
        first_lineno=target.__code__.co_firstlineno,
        filename=filename,
    )
    params = _params_of(tree, name)
    docstring = ast.get_docstring(tree) if isinstance(tree, ast.FunctionDef) else None

    info = FunctionInfo(
        name=name,
        filename=filename or "<unknown>",
        source=textwrap.dedent(raw).strip(),
        tree=tree,
        params=params,
        docstring=docstring,
    )
    _classify_closures(target, info)

    info.user_names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    info.user_names |= {p.name for p in params}
    info.user_names |= set(info.closure_scalars) | set(info.closure_arrays)
    info.user_names |= set(info.math_funcs) | set(info.math_modules) | set(info.user_funcs)
    return info
