# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Reproduce the call-time dtype findings from the 2026-10-23 review.

Each check states the scalar-Python contract and reports ``DEFECT`` when the
vectorized result differs. Run from the repository root:

    uv run python reviews/artifacts/repro_dtype_contract.py

The script exits nonzero while any defect remains; it becomes a regression
harness once the findings are fixed.
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


def main() -> int:
    """Run every dtype-contract reproduction."""
    bools = np.asarray([True, True, False])
    others = np.asarray([True, False, True])

    check("bool + bool is integer", "def f(x, y):\n    return x + y\n", (bools, others), [2, 1, 1])
    check(
        "~bool is integer complement",
        "def f(x):\n    return ~x\n",
        (np.asarray([True, False]),),
        [-2, -1],
    )
    check(
        "abs(bool) is an int",
        "def f(x):\n    return abs(x) + abs(x)\n",
        (np.asarray([True, False]),),
        [2, 0],
    )

    uint_max = np.asarray([2**63 + 3], dtype=np.uint64)
    check(
        "uint64-int64 subtraction is exact",
        "def f(x, y):\n    return x - y\n",
        (uint_max, np.asarray([1], dtype=np.int64)),
        [2**63 + 2],
    )
    check(
        "uint64-int64 remainder is exact",
        "def f(x, y):\n    return x % y\n",
        (np.asarray([2**64 - 1], dtype=np.uint64), np.asarray([3], dtype=np.int64)),
        [0],
    )
    check(
        "uint64-int64 bitwise does not raise",
        "def f(x, y):\n    return x & y\n",
        (uint_max, np.asarray([3], dtype=np.int64)),
        [1],
    )

    check(
        "uint64 + negative literal (per-lane negative result)",
        "def f(x):\n    a = min(x, True)\n    return a + -1\n",
        (np.asarray([0], dtype=np.uint64),),
        [-1],
    )
    check(
        "uint64 * negative literal (per-lane negative result)",
        "def f(x):\n    a = min(x, True)\n    return a * -1\n",
        (np.asarray([1], dtype=np.uint64),),
        [-1],
    )

    c = np.asarray([True, False])
    x = np.asarray([2**63 + 1] * 2, dtype=np.uint64)
    y = np.asarray([-5, -5], dtype=np.int64)
    check(
        "if/else merge keeps uint64/int64 lanes exact",
        "def f(c, x, y):\n    if c > 0:\n        r = x\n    else:\n        r = y\n    return r\n",
        (c, x, y),
        [2**63 + 1, -5],
    )
    check(
        "ternary merge keeps uint64/int64 lanes exact",
        "def f(c, x, y):\n    return x if c > 0 else y\n",
        (c, x, y),
        [2**63 + 1, -5],
    )

    check(
        "abs(int64 minimum) is exact",
        "def f(x):\n    return abs(x)\n",
        (np.asarray([-(2**63)], dtype=np.int64),),
        [2**63],
    )
    check(
        "math.floor of a large int is exact",
        "import math\ndef f(x):\n    return math.floor(x)\n",
        (np.asarray([2**53 + 1], dtype=np.int64),),
        [2**53 + 1],
    )
    check(
        "int(uint64 above int64 max) is exact",
        "def f(x):\n    return int(x)\n",
        (np.asarray([2**63 + 1], dtype=np.uint64),),
        [2**63 + 1],
    )
    check(
        "int ** -1 is true division",
        "def f(x):\n    return x ** -1\n",
        (np.asarray([2], dtype=np.int64),),
        [0.5],
    )

    # Direct evidence for the literal escape: the constant itself is emitted,
    # and adding it to an array silently rounds through float64.
    huge = load("def f(x):\n    return x + int(1e30)\n")
    record(
        "int(1e30) + array must not round",
        call(array_vectorize.vectorize(huge.f), np.asarray([1.0])),
        ("ok", [1 + int(1e30)]),
    )
    record(
        "int(1e30) literal stays within int64",
        "1000000000000000019884624838656" in array_vectorize.vectorize(huge.f).source,
        False,
    )
    record(
        "defaults are range-checked",
        _default_is_rejected(),
        True,
    )

    print()
    if _FAILURES[0]:
        print(f"{_FAILURES[0]} dtype-contract defect(s) reproduced")
        return 1
    print("no dtype-contract defects reproduced")
    return 0


def _default_is_rejected() -> bool:
    """True when a huge literal default is rejected like an over-range body."""
    module = load("def f(x, K=1000000000000000000000000000000):\n    return x + K\n")
    try:
        array_vectorize.vectorize(module.f)
    except Exception:  # noqa: BLE001 - any positioned rejection is acceptable
        return True
    return False


if __name__ == "__main__":
    sys.exit(main())
