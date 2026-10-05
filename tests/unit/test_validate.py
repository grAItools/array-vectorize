"""Validator tests: accept/reject per construct, linter-style diagnostics (T1, T7)."""

from __future__ import annotations

import ast
import importlib.util
import itertools
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from vectorizer import vectorize
from vectorizer._errors import VectorizationError
from vectorizer._extract import extract_function
from vectorizer._validate import _EXPR_MESSAGES, validate

_tmp = tempfile.TemporaryDirectory(prefix="vec_validate_")
_TMPDIR = Path(_tmp.name)
_seq = itertools.count()


def make_fn(body: str, signature: str = "x", type_params: str = "") -> Callable[..., Any]:
    """Define a function with the given (indented) body in a real temp module."""
    path = _TMPDIR / f"snippet_{next(_seq)}.py"
    path.write_text(
        "import math\n\n\ndef subject" + type_params + "(" + signature + "):\n" + body + "\n"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.subject


def check(src: str) -> None:
    validate(extract_function(make_fn(f"    return {src}")))


def check_body(body: str) -> None:
    validate(extract_function(make_fn(body)))


# ------------------------------------------------------------------ accepted


@pytest.mark.parametrize(
    "src",
    [
        "x + 1",
        "x - 1.5 * x / 2",
        "x ** 2",
        "x // 2",
        "x % 2",
        "-x",
        "+x",
        "x & 1",
        "x | 2",
        "x ^ 3",
        "x << 1",
        "x >> 1",
        "~x",
        "abs(x)",
        "min(x, 2)",
        "max(x, 2, 3)",
        "round(x)",
        "int(x)",
        "float(x)",
        "bool(x)",
        "math.sqrt(x)",
        "math.exp(x) + math.log(x)",
        "math.atan2(x, 2.0)",
        "math.pow(x, 2)",
        "math.trunc(x)",
        "math.pi * x",
        "math.inf",
        "math.isnan(x)",
        "x == 1",
        "x != 1",
        "x < 1",
        "x <= 1",
        "x > 1",
        "x >= 1",
        "0 < x < 1",
        "x and x > 0",
        "x or 0.0",
        "not (x > 0)",
        "x if x > 0 else 0.0",
    ],
)
def test_accepted_expressions(src: str) -> None:
    check(src)


def test_accepted_statements() -> None:
    check_body("    y = x + 1\n    return y")
    check_body("    y = x\n    y += 1\n    return y")
    check_body("    y = x\n    y **= 2\n    return y")
    check_body('    """doc"""\n    return x')
    check_body("    pass\n    return x")
    check_body("    if x > 0:\n        return x\n    return 0.0")
    check_body("    if x > 0:\n        y = x\n    else:\n        y = 0.0\n    return y")
    body = (
        "    if x > 0:\n        y = 1\n"
        "    elif x < -1:\n        y = 2\n"
        "    else:\n        y = 3\n"
        "    return y"
    )
    check_body(body)
    check_body("    t = 0.0\n    for i in range(3):\n        t = t + x\n    return t")
    check_body("    t = 0.0\n    for i in range(1, 4):\n        t = t + x\n    return t")


# ------------------------------------------------------------------ rejected


@pytest.mark.parametrize(
    ("src", "msg"),
    [
        ("x @ x", "MatMult"),
        ("'a'", "literals"),
        ("None", "literals"),
        ("1j", "literals"),
        ("b'b'", "literals"),
        ("x in (1, 2)", "tuples"),
        ("x is None", "comparison Is"),
        ("x not in [1]", "lists"),
        ("x[0]", "subscripts"),
        ("(x, 1)", "tuples"),
        ("[x, 1]", "lists"),
        ("{x, 1}", "sets"),
        ("{'a': x}", "dicts"),
        ("[x for x in x]", "comprehensions"),
        ("(lambda y: y)(x)", "lambdas"),
        ("f'{x}'", "f-strings"),
        ("(y := x)", "walrus"),
    ],
)
def test_rejected_expressions(src: str, msg: str) -> None:
    with pytest.raises(VectorizationError, match=msg):
        check(src)


@pytest.mark.parametrize(
    ("body", "msg"),
    [
        ("    while x > 0:\n        x = x - 1\n    return x", "while"),
        ("    for i in range(3):\n        break\n    return x", "break"),
        ("    for i in range(3):\n        continue\n    return x", "continue"),
        ("    global g\n    return x", "global"),
        ("    try:\n        y = x\n    except Exception:\n        y = 0\n    return y", "try"),
        ("    import os\n    return x", "imports"),
        ("    from math import sqrt\n    return x", "imports"),
        ("    with open('f') as fh:\n        y = x\n    return y", "with"),
        ("    match x:\n        case 1:\n            y = 1\n    return y", "match"),
        ("    y: float = x\n    return y", "annotated"),
        ("    y = x\n    x = 2\n    return y", None),  # control: passes
        ("    return x\n    y = 1", "after return"),
        ("    return", "bare returns"),
        ("    return x, 1", "tuples"),
        ("    y = x, 1\n    return y", "tuples"),
        ("    (y, z) = (x, 1)\n    return y", "tuple/list assignment"),
        ("    y = [1]\n    return y", "lists"),
        ("    x[0] = 1\n    return x", "subscript assignment"),
        ("    y = x\n    del y\n    return x", "del"),
        ("    pass", "no return statement"),
        ("    for i in x:\n        y = i\n    return y", "range"),
        ("    for i in range(1, 2, 3, 4):\n        y = i\n    return y", "range"),
        ("    for i in range(n=3):\n        y = i\n    return y", "range"),
        ("    assert x > 0\n    return x", "assert"),
        ("    print(x)\n    return x", "expression statements"),
        ("    y = 1\n    'not a docstring'\n    return y", "expression statements"),
    ],
)
def test_rejected_statements(body: str, msg: str | None) -> None:
    if msg is None:
        check_body(body)
        return
    with pytest.raises(VectorizationError, match=msg):
        check_body(body)


def test_math_attribute_not_in_subset() -> None:
    with pytest.raises(VectorizationError, match="supported math subset"):
        check("math.fsum([x])")


def test_diagnostics_have_positions_and_caret() -> None:
    def subject(x: float) -> float:
        while x > 0:
            x = x - 1
        return x

    with pytest.raises(VectorizationError) as exc:
        validate(extract_function(subject))
    err = exc.value
    assert err.diagnostics, "expected at least one diagnostic"
    d = err.diagnostics[0]
    assert d.lineno == 2
    assert d.col_offset == 4
    assert "while x > 0:" in d.line
    assert "^" in d.render()


def test_all_violations_collected() -> None:
    def subject(x: float) -> float:
        y = x[0]
        while x > 0:
            x = x - 1
        return x, y

    with pytest.raises(VectorizationError) as exc:
        validate(extract_function(subject))
    messages = [d.message for d in exc.value.diagnostics]
    assert any("subscripts" in m for m in messages)
    assert any("while" in m for m in messages)
    assert any("tuples" in m for m in messages)


def test_keyword_args_in_calls_rejected() -> None:
    with pytest.raises(VectorizationError, match="keyword arguments in calls"):
        check("math.copysign(x, y=2.0)")


def test_lambda_direct_is_accepted() -> None:
    info = extract_function(lambda x: x + 1.0)  # type: ignore[arg-type]
    validate(info)


def test_docstring_only_first() -> None:
    check_body('    """doc"""\n    return x')
    with pytest.raises(VectorizationError, match="docstring"):
        check_body('    y = x\n    """not first"""\n    return y')


def test_for_target_must_be_name() -> None:
    with pytest.raises(VectorizationError, match="single name"):
        check_body("    for i, j in range(3):\n        pass\n    return x")


def test_nonlocal_rejected() -> None:
    path = _TMPDIR / "nonlocal_snippet.py"
    path.write_text(
        "def outer():\n"
        "    g = 1.0\n"
        "    def subject(x):\n"
        "        nonlocal g\n"
        "        return x\n"
        "    return subject\n"
    )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(VectorizationError, match="nonlocal"):
        validate(extract_function(mod.outer()))


# ------------------------------------------- Python 3.11-3.14 new syntax
#
# The floor bump widens the input language: the contract is rejection with a
# precise diagnostic, never a silent miscompile (RESTRUCTURE.md §7.1). Each
# test runs the real grammar on the interpreter that supports it; the CI
# matrix (3.12/3.13/3.14) makes the skips meaningful.


def test_try_star_rejected() -> None:
    # except* is valid syntax on every supported interpreter (3.11+)
    with pytest.raises(VectorizationError, match=r"except\*"):
        check_body("    try:\n        y = x\n    except* ValueError:\n        y = 0\n    return y")


def test_type_alias_statement_rejected() -> None:
    # `type X = ...` (PEP 695) is valid syntax on every supported interpreter
    with pytest.raises(VectorizationError, match="type alias statements"):
        check_body("    type Alias = int\n    return x")


def test_generic_function_rejected() -> None:
    # def subject[T](x) (PEP 695): type parameters on the extracted function
    with pytest.raises(VectorizationError, match="type parameters"):
        validate(extract_function(make_fn("    return x", type_params="[T]")))


def test_template_string_table_entry_is_version_gated() -> None:
    # the t-string rejection entry exists exactly where the grammar has
    # ast.TemplateStr (3.14+); on 3.12/3.13 the guard must keep it absent
    template_node = getattr(ast, "TemplateStr", None)
    if sys.version_info >= (3, 14):
        assert template_node is not None
        assert template_node in _EXPR_MESSAGES
    else:
        assert template_node is None
        assert template_node not in _EXPR_MESSAGES


@pytest.mark.skipif(sys.version_info < (3, 14), reason="t-string syntax requires 3.14")
def test_template_string_rejected() -> None:
    # t-strings (PEP 750) only exist in the 3.14 grammar; on 3.12/3.13 the
    # source itself would be a SyntaxError, so the rejection is only testable
    # here (the validator's guarded message-table entry covers both)
    with pytest.raises(VectorizationError, match="t-strings"):
        check('t"x"')


# ------------------------------------- targeted validator error branches
# (merged from test_validate_branches.py; make_fn takes the original
# "x, y=2.0" signature via keyword)


@pytest.mark.parametrize(
    ("body", "msg"),
    [
        ("    x //= 2\n    return x", None),  # control: valid augassign
        ("    x @= 2\n    return x", "augmented assignment"),
        ("    x = y = 1\n    return x", "multiple assignment targets"),
        ("    x.foo = 1\n    return x", "attribute assignment"),
        ("    del x\n    return x", None if False else "not supported"),
        ("    for i in range(3):\n        pass\n    else:\n        pass\n    return x", "for/else"),
        ("    return x + 9223372036854775808", "int64 range"),
        ("    return x", None),
    ],
)
def test_validator_branches(body: str, msg: str | None) -> None:
    if msg is None:
        validate(extract_function(make_fn(body, signature="x, y=2.0")))
        return
    with pytest.raises(VectorizationError, match=msg):
        validate(extract_function(make_fn(body, signature="x, y=2.0")))


def test_attribute_assignment() -> None:
    with pytest.raises(VectorizationError, match="attribute assignment"):
        validate(
            extract_function(make_fn("    a = x\n    a.f = 1\n    return a", signature="x, y=2.0"))
        )


def test_generic_attribute_access_rejected() -> None:
    with pytest.raises(VectorizationError, match="attribute access"):
        validate(extract_function(make_fn("    return x.real", signature="x, y=2.0")))


def test_starred_call_argument_rejected() -> None:
    with pytest.raises(VectorizationError, match="starred call arguments"):
        validate(extract_function(make_fn("    return math.hypot(*[x, y])", signature="x, y=2.0")))


def test_unknown_expression_falls_through() -> None:
    # Starred in a non-call context hits the generic expression message
    with pytest.raises(VectorizationError, match="not supported"):
        validate(extract_function(make_fn("    return x + (y := 1)", signature="x, y=2.0")))


def test_del_is_rejected_by_vectorize() -> None:
    with pytest.raises(VectorizationError, match="statements are not supported"):
        vectorize(make_fn("    del x\n    return 1.0", signature="x, y=2.0"))
