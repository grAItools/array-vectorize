# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Reproduce the strict-backend neutrality findings from the 2026-10-23 review.

Run from the repository root:

    uv run python reviews/artifacts/repro_backend_neutrality.py

Every check calls the generated function with ``array_api_strict`` arrays.
The script exits nonzero while any defect remains.
"""

from __future__ import annotations

import importlib.util
import itertools
import pathlib
import sys
import tempfile
from typing import Any

import numpy as np

import array_vectorize

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


def call(fn: Any, *args: Any) -> tuple[str, Any]:
    """Return ``("ok", value)`` or ``("error", "ExcType: message")``."""
    try:
        return "ok", np.asarray(fn(*args)).tolist()
    except Exception as exc:  # noqa: BLE001 - the exception text is the evidence
        return "error", f"{type(exc).__name__}: {exc}"


def record(label: str, actual: Any, expected: Any) -> None:
    """Print one contract comparison and count failures."""
    ok = actual == expected
    if not ok:
        _FAILURES[0] += 1
    print(f"[{'ok' if ok else 'DEFECT'}] {label}")
    print(f"    actual:   {actual}")
    print(f"    expected: {expected}")


def check(label: str, source: str, args: tuple[Any, ...], expected: Any) -> None:
    """Vectorize ``source`` and compare its result with ``expected``."""
    fn = load(source).f
    record(label, call(array_vectorize.vectorize(fn), *args), ("ok", expected))


def nan_result(fn: Any, arg: Any) -> bool:
    """True when every lane is NaN."""
    try:
        return bool(np.all(np.isnan(np.asarray(fn(arg), dtype=float))))
    except Exception:  # noqa: BLE001 - an exception is not a NaN result
        return False


def inf_result(fn: Any, arg: Any) -> bool:
    """True when every lane is an infinity."""
    try:
        return bool(np.all(np.isinf(np.asarray(fn(arg), dtype=float))))
    except Exception:  # noqa: BLE001 - an exception is not an inf result
        return False


def main() -> int:
    """Run every backend-neutrality reproduction."""
    try:
        import array_api_strict as xps
    except ImportError:
        print("array-api-strict is not installed; run `uv sync` first")
        return 0

    xp = xps
    check(
        "loop-index xp calls accept Python ints",
        "def f(x):\n"
        "    s = 0.0\n"
        "    for i in range(3):\n"
        "        s = s + (i // 2) + (i % 2) + (i ** 2)\n"
        "    return s + x\n",
        (xp.asarray([0.0]),),
        [7.0],
    )
    check(
        "scalar flag condition reaches xp.where as an array",
        "def f(x, flag):\n    if flag > 0:\n        return x\n    return -x\n",
        (xp.asarray([1.0, 2.0]), 1.0),
        [1.0, 2.0],
    )
    check(
        "not numeric is x == 0",
        "def f(x):\n    return not x\n",
        (xp.asarray([0.0, 2.0]),),
        [True, False],
    )
    check(
        "bool + bool is integer",
        "def f(x, y):\n    return x + y\n",
        (xp.asarray([True, True]), xp.asarray([True, False])),
        [2, 1],
    )
    closure = load(
        "import numpy as np\nARR = np.asarray([1.0, 2.0])\n\n\ndef f(x):\n    return x + ARR\n"
    )
    record(
        "captured numpy array works with another backend",
        call(array_vectorize.vectorize(closure.f), xp.asarray([1.0, 2.0])),
        ("ok", [2.0, 4.0]),
    )
    captured_scalar = load("import numpy as np\nK = np.int64(3)\n\n\ndef f(x):\n    return x * K\n")
    record(
        "captured NumPy scalar works with another backend",
        call(array_vectorize.vectorize(captured_scalar.f), xp.asarray([1.0, 2.0])),
        ("ok", [3.0, 6.0]),
    )

    modulo = array_vectorize.vectorize(load("def f(x):\n    return x % 0\n").f)
    record(
        "integer x % 0 is NaN as documented",
        nan_result(modulo, xp.asarray([4], dtype=xp.int64)),
        True,
    )
    floordiv = array_vectorize.vectorize(load("def f(x):\n    return x // 0\n").f)
    record(
        "integer x // 0 is inf as documented",
        inf_result(floordiv, xp.asarray([4], dtype=xp.int64)),
        True,
    )

    print()
    if _FAILURES[0]:
        print(f"{_FAILURES[0]} backend-neutrality defect(s) reproduced")
        return 1
    print("no backend-neutrality defects reproduced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
