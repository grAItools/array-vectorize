"""Tests for source extraction and closure capture."""

from __future__ import annotations

import math

import numpy as np
import pytest
from support import make_module

from array_vectorize.errors import VectorizationError
from array_vectorize.frontend.extract import extract_function

K = 2.5
ARR = np.asarray([1.0, 2.0])
STR_GLOBAL = "nope"


def fn_plain(x: float) -> float:
    """Docstring here."""
    return x * K


def test_plain_function() -> None:
    info = extract_function(fn_plain)
    assert info.name == "fn_plain"
    assert info.params[0].name == "x"
    assert info.docstring == "Docstring here."
    assert info.closure_scalars == {"K": 2.5}
    assert "def fn_plain" in info.source


def test_lambda_via_assignment() -> None:
    lam = extract_function(lambda x: x + 1.0)  # type: ignore[arg-type]
    assert lam.params[0].name == "x"
    assert lam.docstring is None


def test_decorated_source_decorator_ignored() -> None:
    def deco(f):
        return f

    @deco
    def fn(x: float) -> float:
        return x

    info = extract_function(fn)
    assert info.params[0].name == "x"


def test_unwrap_follows_wrapped() -> None:
    import functools

    @functools.wraps(fn_plain)
    def wrapper(x: float) -> float:
        raise AssertionError("never called")

    info = extract_function(wrapper)
    assert info.name == "fn_plain"


def test_resolve_original_follows_marker_chains() -> None:
    import functools

    from array_vectorize import vectorize
    from array_vectorize.frontend.extract import resolve_original

    @functools.wraps(fn_plain)
    def wrapper(x: float) -> float:
        raise AssertionError("never called")

    # __wrapped__ chains resolve to the innermost function...
    assert resolve_original(wrapper) is fn_plain
    # ...and prior vectorizations chain to the scalar original
    vec = vectorize(fn_plain)
    assert resolve_original(vec) is fn_plain
    assert resolve_original(vectorize(vec)) is fn_plain
    # plain functions pass through unchanged
    assert resolve_original(fn_plain) is fn_plain


def test_builtin_rejected() -> None:
    with pytest.raises(VectorizationError, match="builtin"):
        extract_function(math.sqrt)  # type: ignore[arg-type]


def test_method_rejected() -> None:
    class C:
        def m(self, x: float) -> float:
            return x

    with pytest.raises(VectorizationError, match="plain Python function"):
        extract_function(C().m)  # type: ignore[arg-type]


def test_no_source_rejected_with_guidance() -> None:
    ns: dict[str, object] = {}
    exec("def f(x): return x", ns)
    with pytest.raises(VectorizationError, match="REPL"):
        extract_function(ns["f"])  # type: ignore[arg-type]


def test_vararg_rejected() -> None:
    def f(*args: float) -> float:
        return 0.0

    with pytest.raises(VectorizationError, match=r"\*args"):
        extract_function(f)


def test_kwarg_rejected() -> None:
    def f(x: float, **kw: float) -> float:
        return x

    with pytest.raises(VectorizationError, match=r"\*\*kwargs"):
        extract_function(f)


def test_non_literal_default_rejected() -> None:
    def f(x: float, s: str = "a") -> float:
        return x

    with pytest.raises(VectorizationError, match="literal"):
        extract_function(f)


def test_none_default_rejected() -> None:
    def f(x: float, y: float | None = None) -> float:  # type: ignore[assignment]
        return x

    with pytest.raises(VectorizationError, match="literal"):
        extract_function(f)


def test_zero_params_rejected() -> None:
    def f() -> float:
        return 1.0

    with pytest.raises(VectorizationError, match="zero-argument"):
        extract_function(f)


def test_literal_defaults_recorded() -> None:
    def f(x: float, scale: float = 2.0, n: int = 3, flag: bool = True) -> float:
        return x * scale

    info = extract_function(f)
    assert [(p.name, p.default, p.has_default) for p in info.params[1:]] == [
        ("scale", 2.0, True),
        ("n", 3, True),
        ("flag", True, True),
    ]


def test_kwonly_params() -> None:
    def f(x: float, *, scale: float = 1.0) -> float:
        return x * scale

    info = extract_function(f)
    assert info.params[1].kind == "kwonly"
    assert info.params[1].default == 1.0


def test_from_math_import_resolved_by_identity() -> None:
    from math import sqrt

    def f(x: float) -> float:
        return sqrt(x)  # type: ignore[misc]

    info = extract_function(f)
    assert info.math_funcs == {"sqrt": "sqrt"}


def test_math_module_recorded() -> None:
    def f(x: float) -> float:
        return math.exp(x)

    info = extract_function(f)
    assert info.math_modules == {"math"}


def test_closure_array_captured() -> None:
    def f(x: float) -> float:
        return x + ARR[0] * 0.0 + x * 0.0 + x * 1.0  # ARR referenced (subscript rejected later)

    info = extract_function(f)
    assert info.closure_arrays == {"ARR": ARR}


def test_unresolvable_global_rejected() -> None:
    def f(x: float) -> float:
        return x + len(STR_GLOBAL) * 0.0  # type: ignore[misc]

    with pytest.raises(VectorizationError, match="unsupported type"):
        extract_function(f)


def test_unbound_name_rejected() -> None:
    def f(x: float) -> float:
        return x + unknown_name  # type: ignore[name-defined] # noqa: F821

    with pytest.raises(VectorizationError, match="not resolvable"):
        extract_function(f)


def test_user_function_captured() -> None:
    def helper(y: float) -> float:
        return y * 2.0

    def f(x: float) -> float:
        return helper(x)  # type: ignore[misc]

    info = extract_function(f)
    assert info.user_funcs == {"helper": helper}


def test_user_names_collects_source_names() -> None:
    def f(x: float, x_1: float) -> float:
        y = x + x_1
        return y

    info = extract_function(f)
    assert {"x", "x_1", "y"} <= info.user_names


def test_np_float64_is_scalar() -> None:
    c = np.float64(1.5)

    def f(x: float) -> float:
        return x * c

    info = extract_function(f)
    assert info.closure_scalars == {"c": 1.5}


def test_lambda_probe_fallback_identifies_sibling() -> None:
    # module source unavailable (or uncompilable): the isolated-probe
    # comparison must still identify the right same-line lambda
    import ast
    import importlib.util
    import inspect
    import pathlib
    import tempfile

    from array_vectorize.frontend.lambda_id import _find_target

    tmp = pathlib.Path(tempfile.mkdtemp())
    path = tmp / "probe_siblings.py"
    path.write_text("f1, f2 = (lambda x: x + 1), (lambda x: x + 2)\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for fn, body in ((mod.f1, "lambda x: x + 1"), (mod.f2, "lambda x: x + 2")):
        tree = ast.parse(inspect.getsource(fn))
        node = _find_target(tree, "<lambda>", fn, module_source=None)
        assert ast.unparse(node) == body

    # a module source that cannot be compiled also falls back to probes
    tree = ast.parse(inspect.getsource(mod.f2))
    node = _find_target(tree, "<lambda>", mod.f2, module_source="def broken(:", first_lineno=1)
    assert ast.unparse(node) == "lambda x: x + 2"


def test_lambda_probe_fallback_interchangeable() -> None:
    import ast
    import importlib.util
    import inspect
    import pathlib
    import tempfile

    from array_vectorize.frontend.lambda_id import _find_target

    tmp = pathlib.Path(tempfile.mkdtemp())
    path = tmp / "probe_twin.py"
    path.write_text("f1, f2 = (lambda x: x + 1), (lambda x: x + 1)\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    tree = ast.parse(inspect.getsource(mod.f2))
    node = _find_target(tree, "<lambda>", mod.f2, module_source=None)
    assert ast.unparse(node) == "lambda x: x + 1"


def test_lambda_probe_fallback_rejects_unidentifiable() -> None:
    import ast
    import importlib.util
    import inspect
    import pathlib
    import tempfile

    from array_vectorize.frontend.lambda_id import _find_target

    tmp = pathlib.Path(tempfile.mkdtemp())
    path = tmp / "probe_none.py"
    path.write_text("f1, f2 = (lambda x: x + 1), (lambda x: x + 2)\nf3 = lambda x: x + 3\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    tree = ast.parse(inspect.getsource(mod.f1))
    with pytest.raises(VectorizationError, match="could not be identified"):
        _find_target(tree, "<lambda>", mod.f3, module_source=None)


def test_async_function_rejected() -> None:
    mod = make_module("async def subject(x):\n    return x\n")
    with pytest.raises(VectorizationError, match="async"):
        extract_function(mod.subject)


def test_kwonly_without_default() -> None:
    mod = make_module("def subject(x, *, scale):\n    return x * scale\n")
    info = extract_function(mod.subject)
    assert info.params[1].kind == "kwonly"
    assert not info.params[1].has_default


def test_expr_lambda_top_level() -> None:
    make_module("(lambda x: x + 1)\n")
    # nothing to grab; just ensure parse path works via direct lambda
    info = extract_function(lambda x: x + 1.0)  # type: ignore[arg-type]
    assert info.params[0].name == "x"


def test_no_target_in_source_rejected() -> None:
    mod = make_module("X = 1\n")
    with pytest.raises(VectorizationError):
        extract_function(mod.X)  # type: ignore[arg-type]
