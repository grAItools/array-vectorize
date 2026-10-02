"""Source extraction and closure/constant capture (plan §4)."""

from __future__ import annotations

import ast
import inspect
import math
import textwrap
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NoReturn

from ._builtins import MATH_FUNCS, MATH_SPECIAL
from ._errors import VectorizationError

__all__ = ["FunctionInfo", "Param", "extract_function"]

_AstFunction = ast.FunctionDef | ast.Lambda

_MATH_FUNC_NAMES = (*MATH_FUNCS, *MATH_SPECIAL)


@dataclass(frozen=True)
class Param:
    """One function parameter; ``kind`` mirrors Python argument kinds."""

    name: str
    kind: str  # 'posonly' | 'arg' | 'kwonly'
    default: int | float | bool | None = None
    has_default: bool = False


@dataclass
class FunctionInfo:
    """Everything the later passes need about the scalar function."""

    name: str
    filename: str
    source: str  # dedented original source text (embedded verbatim in the docstring)
    tree: _AstFunction
    params: list[Param]
    docstring: str | None
    closure_scalars: dict[str, int | float | bool] = field(default_factory=dict)
    closure_arrays: dict[str, Any] = field(default_factory=dict)
    math_funcs: dict[str, str] = field(default_factory=dict)  # name -> math attr
    math_modules: set[str] = field(default_factory=set)  # names bound to `math`
    user_funcs: dict[str, Callable[..., Any]] = field(default_factory=dict)
    user_names: set[str] = field(default_factory=set)

    @property
    def param_names(self) -> list[str]:
        return [p.name for p in self.params]


def _reject(name: str, message: str) -> NoReturn:
    raise VectorizationError(f"cannot vectorize {name!r}: {message}")


def _find_target(tree: ast.Module, name: str, target: object) -> _AstFunction:
    for stmt in tree.body:
        if isinstance(stmt, ast.AsyncFunctionDef):
            _reject(name, "async functions are not supported")
        if isinstance(stmt, ast.FunctionDef | ast.Lambda):
            return stmt
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Lambda)
        ):
            return stmt.value
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Lambda):
            return stmt.value
    # A lambda passed directly, e.g. vectorize(lambda x: x + 1): its source
    # line contains the lambda nested in an expression. When several lambdas
    # share the line, identify the right one by comparing compiled bytecode
    # (code, constants, AND referenced names) plus default values with the
    # target's code object; reject when identification is ambiguous.
    candidates = [n for n in ast.walk(tree) if isinstance(n, ast.Lambda)]
    if not candidates:
        _reject(name, "could not locate the function definition in its source")
    if len(candidates) == 1:
        return candidates[0]
    code = getattr(target, "__code__", None)
    if code is None:
        return candidates[0]
    matches: list[ast.Lambda] = []
    for node in candidates:
        # NOTE: no line-number pre-filter: getsource returns a block whose
        # linenos are relative to the block, not the original file
        try:
            probe = compile(ast.Expression(node), "<lambda-probe>", "eval")
        except SyntaxError:
            continue
        # the expression wraps the lambda in MAKE_FUNCTION; the lambda's own
        # code object is nested in the probe's constants
        inner = next((c for c in probe.co_consts if isinstance(c, types.CodeType)), None)
        if inner is None:
            continue
        if (
            inner.co_code != code.co_code
            or _const_key(inner.co_consts) != _const_key(code.co_consts)
            or inner.co_names != code.co_names
            or inner.co_varnames != code.co_varnames
        ):
            continue
        # compare default values: bytecode alone cannot distinguish
        # `lambda x, y=1: ...` from `lambda x, y=2: ...` (defaults live on
        # the function object, not the code object)
        sentinel = object()
        ast_defaults = tuple(
            d.value if isinstance(d, ast.Constant) else sentinel for d in node.args.defaults
        )
        ast_kw_defaults = tuple(
            d.value if isinstance(d, ast.Constant) else sentinel
            for d in node.args.kw_defaults
            if d is not None
        )
        fn_defaults = tuple(getattr(target, "__defaults__", None) or ())
        fn_kw_defaults = tuple(
            v for k, v in (getattr(target, "__kwdefaults__", None) or {}).items()
        )
        if _const_key(ast_defaults) != _const_key(fn_defaults) or _const_key(
            ast_kw_defaults
        ) != _const_key(fn_kw_defaults):
            continue
        matches.append(node)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        _reject(
            name,
            "multiple lambdas share this source line and the intended one "
            "could not be identified; assign the lambda to a variable on its "
            "own line",
        )
    # several identical lambdas: semantically interchangeable
    return matches[0]


def _const_key(value: Any) -> Any:
    """Type- and sign-precise comparison key for constants.

    Plain equality merges values that are semantically distinct: ``0.0 ==
    -0.0`` (copysign differs), ``True == 1``, and ``1 == 1.0``. Key each
    constant by its exact type, and zero floats by their sign bit. Recurses
    into tuples (co_consts nests constants in tuples).
    """
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        if value == 0.0:
            return ("zero-float", math.copysign(1.0, value) < 0)
        return ("float", value)
    if isinstance(value, tuple):
        return tuple(_const_key(v) for v in value)
    return value


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
        kind = "posonly" if i < len(a.posonlyargs) else "arg"
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
        elif hasattr(value, "__array_namespace__"):
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

    tree = _find_target(ast.parse(textwrap.dedent(raw)), name, target)
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
