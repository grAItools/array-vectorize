"""Lambda identification subsystem: pair a function object with its AST node.

Grew out of the review-round fix ``5ebff7d`` (raw ``co_code`` is not stable
across compilation units). Two-strategy matcher: recompile the target's
module and pair code objects with AST nodes by line/order; fall back to
isolated-probe bytecode comparison.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
import math
import types
from typing import Any

from array_vectorize import errors
from array_vectorize.frontend import info


def _lambdas_in_source_order(tree: ast.AST) -> Iterator[ast.Lambda]:
    """All lambda nodes below ``tree`` in DFS pre-order (= source order).

    The compiler emits nested code objects in the same order, so this
    ordering pairs 1:1 with ``_iter_code_objects`` on the same source.
    """
    for child in ast.iter_child_nodes(tree):
        if isinstance(child, ast.Lambda):
            yield child
        yield from _lambdas_in_source_order(child)


def _iter_code_objects(code: types.CodeType) -> Iterator[types.CodeType]:
    """All code objects nested in ``code``'s constants, in source order."""
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield const
            yield from _iter_code_objects(const)


def _code_key(code: types.CodeType) -> Any:
    """Value-based identity key for a code object.

    Code objects compare by identity, so nested code objects in
    ``co_consts`` must be keyed recursively (plain constants via
    ``_const_key``).
    """
    return (
        code.co_code,
        code.co_names,
        code.co_varnames,
        tuple(
            _code_key(c) if isinstance(c, types.CodeType) else _const_key(c) for c in code.co_consts
        ),
    )


def _defaults_match(node: ast.Lambda, target: object) -> bool:
    """Compare a lambda's AST default values with a function object's.

    Bytecode cannot distinguish ``lambda x, y=1: ...`` from
    ``lambda x, y=2: ...``: defaults live on the function object, not
    the code object. Non-constant defaults compare as "different"
    (sentinel) — they are rejected elsewhere anyway.
    """
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
    fn_kw_defaults = tuple(v for k, v in (getattr(target, "__kwdefaults__", None) or {}).items())
    same = _const_key(ast_defaults) == _const_key(fn_defaults) and _const_key(
        ast_kw_defaults
    ) == _const_key(fn_kw_defaults)
    return bool(same)


def _find_target(
    tree: ast.Module,
    name: str,
    target: object,
    module_source: str | None = None,
    first_lineno: int | None = None,
    filename: str | None = None,
) -> info._AstFunction:
    for stmt in tree.body:
        if isinstance(stmt, ast.AsyncFunctionDef):
            errors._reject(name, "async functions are not supported")
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
    # share the line, identify the right one. Primary strategy: recompile the
    # target's own module (byte-identical context — raw co_code is NOT stable
    # across compilation units: a name bound by an `import` in the unit makes
    # the compiler emit LOAD_ATTR+NULL instead of LOAD_METHOD for calls on
    # it, so isolated probes can mismatch a module-compiled twin) and pair
    # the module's code objects with the candidate AST nodes by line and
    # order. Fallback: compile each candidate as an isolated expression and
    # compare bytecode, constants, referenced names, and default values.
    candidates = list(_lambdas_in_source_order(tree))
    if not candidates:
        errors._reject(name, "could not locate the function definition in its source")
    if len(candidates) == 1:
        return candidates[0]
    code = getattr(target, "__code__", None)
    if code is None:
        return candidates[0]
    if module_source is not None and first_lineno is not None:
        # primary: recompile the module and pair by line + source order
        try:
            mod_code = compile(module_source, filename or "<module>", "exec", dont_inherit=True)
        except (SyntaxError, ValueError, TypeError):
            mod_code = None
        if mod_code is not None:
            # block linenos are relative: the block starts at the target's
            # line, so block lineno 1 is the target's line in the file
            at_line = [n for n in candidates if n.lineno == 1]
            module_lambdas = [
                c for c in _iter_code_objects(mod_code) if c.co_firstlineno == first_lineno
            ]
            target_key = _code_key(code)
            for node, twin in zip(at_line, module_lambdas, strict=False):
                # defaults are checked too: bytecode alone cannot
                # distinguish `lambda x, y=1: ...` from `lambda x, y=2: ...`
                if _code_key(twin) == target_key and _defaults_match(node, target):
                    return node
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
        if not _defaults_match(node, target):
            continue
        matches.append(node)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        errors._reject(
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
