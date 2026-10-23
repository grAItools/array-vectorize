# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Reproduce the inspectability, verify, name-hygiene, and decorator findings.

Run from the repository root:

    uv run python reviews/artifacts/repro_emit_verify.py

The script exits nonzero while any defect remains.
"""

from __future__ import annotations

import importlib.util
import inspect
import itertools
import pathlib
import sys
import tempfile
from typing import Any

import numpy as np

import array_vectorize
from array_vectorize import verify

_TMP = tempfile.TemporaryDirectory(prefix="array_vectorize_review_")
_SEQ = itertools.count()
_FAILURES = [0]


def load(source: str) -> Any:
    """Write a real module file and import it (inspect.getsource needs one)."""
    path = pathlib.Path(_TMP.name) / f"subject_{next(_SEQ)}.py"
    path.write_text(source, encoding="utf-8", newline="\n")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(label: str, actual: Any, expected: Any, detail: str = "") -> None:
    """Print one contract comparison and count failures."""
    ok = actual == expected
    if not ok:
        _FAILURES[0] += 1
    print(f"[{'ok' if ok else 'DEFECT'}] {label}")
    if detail:
        print(f"    detail:   {detail}")
    print(f"    actual:   {actual}")
    print(f"    expected: {expected}")


def raises_vectorization_error(action: Any) -> bool:
    """True when ``action()`` raises ``VectorizationError``."""
    try:
        action()
    except array_vectorize.VectorizationError:
        return True
    return False


def main() -> int:
    """Run every reproduction."""
    first = load("def f(x):\n    return x + 1\n")
    second = load("def f(x):\n    return x * 100 + 7\n")
    v1 = array_vectorize.vectorize(first.f)
    v2 = array_vectorize.vectorize(second.f)
    del v2
    source_of_v1 = inspect.getsource(v1)
    record(
        "same-named functions have distinct generated filenames",
        "x * 100 + 7" in source_of_v1,
        False,
        detail=source_of_v1.splitlines()[-1],
    )

    options = load(
        "import math\n\n\ndef f(x):\n    if x > 0:\n        return math.sqrt(x)\n    return 0.0\n"
    )
    plain = array_vectorize.vectorize(options.f)
    protected = array_vectorize.vectorize(options.f, protect_domains=True)
    del protected
    record(
        "option variants do not overwrite the canonical source",
        "logical_and" in inspect.getsource(plain),
        False,
    )

    decorated = load(
        "import functools\n"
        "\n"
        "\n"
        "def deco(fn):\n"
        "    @functools.wraps(fn)\n"
        "    def wrapper(x):\n"
        "        return fn(x) + 100.0\n"
        "    return wrapper\n"
        "\n"
        "\n"
        "@deco\n"
        "def f(x):\n"
        "    return x * 2.0\n"
    )
    vec = array_vectorize.vectorize(decorated.f)
    record(
        "decorated target keeps the wrapper's behavior",
        np.asarray(vec(np.asarray([3.0]))).tolist(),
        [decorated.f(3.0)],
    )

    divide = load("def f(x):\n    return 1.0 / x\n")
    record(
        "verify= accepts the documented 1/0 -> inf divergence",
        raises_vectorization_error(
            lambda: array_vectorize.vectorize(divide.f, verify=(np.asarray([0.0, 2.0]),))
        ),
        False,
    )
    record(
        "verify= detects a large-int off-by-one",
        raises_vectorization_error(
            lambda: verify.verify_match(
                lambda x: x + 1,
                lambda x: x,
                (np.asarray([2**62], dtype=np.int64),),
            )
        ),
        True,
    )

    frozen = load("SCALE = 3.0\n\n\ndef f(x):\n    return x * SCALE\n")
    example = np.asarray([1.0])
    array_vectorize.vectorize(frozen.f, verify=(example,))
    frozen.SCALE = 5.0
    record(
        "verify= on a cache hit follows the frozen environment",
        raises_vectorization_error(lambda: array_vectorize.vectorize(frozen.f, verify=(example,))),
        False,
    )

    shadowed = load(
        "import builtins\n"
        "\n"
        "\n"
        "def myrange(n):\n"
        "    return builtins.range(n + 1)\n"
        "\n"
        "\n"
        "range = myrange\n"
        "\n"
        "\n"
        "def f(x):\n"
        "    s = 0.0\n"
        "    for i in range(2):\n"
        "        s = s + x\n"
        "    return s\n"
    )
    shadowed_vec = array_vectorize.vectorize(shadowed.f)
    record(
        "a shadowed range() is not treated as the builtin",
        np.asarray(shadowed_vec(np.asarray([1.0]))).tolist(),
        [shadowed.f(1.0)],
    )

    reserved = load("def f(array_namespace):\n    return array_namespace + 1\n")
    reserved_vec = array_vectorize.vectorize(reserved.f)
    try:
        reserved_result: Any = np.asarray(reserved_vec(array_namespace=np.asarray([1.0]))).tolist()
    except Exception as exc:  # noqa: BLE001 - the exception text is the evidence
        reserved_result = f"{type(exc).__name__}: {exc}"
    record(
        "reserved parameter names still accept keyword calls",
        reserved_result,
        [2.0],
    )

    growth = load(
        "import math\n"
        "\n"
        "\n"
        "def f(x):\n"
        "    if x > 0:\n"
        "        return math.sqrt(math.log(math.sqrt(x) + 1.0) + 1.0)\n"
        "    return 0.0\n"
    )
    plain_vec: Any = array_vectorize.vectorize(growth.f)
    plain_source: str = plain_vec.source
    protected_vec: Any = array_vectorize.vectorize(growth.f, protect_domains=True)
    protected_source: str = protected_vec.source
    record(
        "protect_domains does not multiply generated code size",
        len(protected_source) <= 2 * len(plain_source),
        True,
        detail=f"plain={len(plain_source)}, protected={len(protected_source)}",
    )

    record(
        "the int64 minimum literal is supported",
        raises_vectorization_error(
            lambda: array_vectorize.vectorize(
                load("def f(x):\n    return x - 9223372036854775808\n").f
            )
        ),
        False,
    )

    print()
    if _FAILURES[0]:
        print(f"{_FAILURES[0]} defect(s) reproduced")
        return 1
    print("no defects reproduced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
