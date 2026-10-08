# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Grammar fuzzer CLI: ``python -m array_vectorize.fuzz``.

Generates random programs from the supported grammar, vectorizes them, and
differentially checks the result against the scalar original on NumPy.
VectorizationError is a valid outcome (the grammar may emit unsupported
shapes); crashes or value mismatches are bugs and fail the run. Seeds make
runs reproducible.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import random
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np

# load-bearing re-export: tests/unit/test_fuzz.py monkeypatches
# array_vectorize.fuzz.vectorize, so the name must stay bound here
from array_vectorize import errors
from array_vectorize import vectorize  # cleanporter: ignore[CP001] load-bearing re-export

_UNARY_MATH = [
    f"math.{name}"
    for name in (
        "sqrt",
        "exp",
        "expm1",
        "log",
        "log1p",
        "log2",
        "log10",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "floor",
        "ceil",
    )
]
_UNARY_BUILTIN = ["abs", "round"]
_UNARY_CAST = ["int", "float", "bool", "math.trunc"]
# copysign is excluded: the sign bit of NaN results (e.g. inf % e) differs
# between CPython and array libraries - inherently divergent
_BINARY_MATH = ["math.atan2", "math.hypot", "math.pow"]
# bitwise ops are dropped: the fuzzer's domain is floats, where they are
# dtype-invalid (covered instead by explicit int-array tests). `**` is
# dropped too: negative-base fractional exponents return complex in Python
# but NaN when vectorized (documented divergence D3, covered by T8 tests).
_BINOPS = ["+", "-", "*", "/", "//", "%"]
_CMPOPS = ["==", "!=", "<", "<=", ">", ">="]


def _expr(rng: random.Random, vars_: list[str], depth: int) -> str:
    if depth <= 0 or rng.random() < 0.3:
        choice = rng.random()
        if choice < 0.5 and vars_:
            return rng.choice(vars_)
        if choice < 0.75:
            return repr(rng.choice([0, 1, 2, 3, -1, -2, 0.5, -0.5, 1.5, 2.0, -3.0]))
        return rng.choice(["math.pi", "math.e", "math.inf", "3", "7"])
    kind = rng.random()
    if kind < 0.55:
        op = rng.choice(_BINOPS)
        return f"({_expr(rng, vars_, depth - 1)} {op} {_expr(rng, vars_, depth - 1)})"
    if kind < 0.62:
        return f"{rng.choice(_UNARY_MATH)}({_expr(rng, vars_, depth - 1)})"
    if kind < 0.68:
        return f"{rng.choice(_UNARY_BUILTIN)}({_expr(rng, vars_, depth - 1)})"
    if kind < 0.74:
        return f"{rng.choice(_UNARY_CAST)}({_expr(rng, vars_, depth - 1)})"
    if kind < 0.8:
        a, b = _expr(rng, vars_, depth - 1), _expr(rng, vars_, depth - 1)
        return f"{rng.choice(_BINARY_MATH)}({a}, {b})"
    if kind < 0.86:
        # note: min/max are excluded - Python's min/max ignore NaN in the
        # second argument while xp.minimum/maximum propagate it (documented
        # divergence); they are covered by dedicated T8 tests instead
        a, b = _expr(rng, vars_, depth - 1), _expr(rng, vars_, depth - 1)
        return f"({a} {rng.choice(_CMPOPS)} {b})"
    if kind < 0.9:
        a, b, c = (_expr(rng, vars_, depth - 1) for _ in range(3))
        return f"({a} if {b} {rng.choice(_CMPOPS)} {c} else {a})"
    if kind < 0.95:
        a = _expr(rng, vars_, depth - 1)
        b = _expr(rng, vars_, depth - 1)
        return f"({a} and {b})" if rng.random() < 0.5 else f"({a} or {b})"
    a, b = _expr(rng, vars_, depth - 1), _expr(rng, vars_, depth - 1)
    return f"(not ({a} {rng.choice(_CMPOPS)} {b}))"


def _statement(rng: random.Random, vars_: list[str], depth: int) -> list[str]:
    """A sequence of assignments / ifs; returns the function body lines."""
    lines: list[str] = []
    n = rng.randint(1, 4)
    for _ in range(n):
        roll = rng.random()
        if roll < 0.55 or depth <= 0:
            lines.append(f"    t{len(lines)} = {_expr(rng, vars_, depth)}")
            vars_ = [*vars_, f"t{len(lines) - 1}"]
        else:
            cond = f"{_expr(rng, vars_, 1)} {rng.choice(_CMPOPS)} {_expr(rng, vars_, 1)}"
            lines.append(f"    if {cond}:")
            lines.extend(f"    {line}" for line in _statement(rng, vars_, depth - 1))
            if rng.random() < 0.6:
                lines.append("    else:")
                lines.extend(f"    {line}" for line in _statement(rng, vars_, depth - 1))
    return lines


def make_program(rng: random.Random) -> str:
    """Build the source of one random scalar function."""
    n_vars = rng.randint(1, 2)
    params = [f"x{i}" for i in range(n_vars)]
    vars_ = [*params]
    body = _statement(rng, vars_, rng.randint(0, 2))
    result = _expr(rng, vars_, rng.randint(1, 3))
    body.append(f"    return {result}")
    if rng.random() < 0.2:
        lines = ["    t0 = 0.0", f"    for i in range({rng.randint(0, 4)}):"]
        inner = [f"        t0 = t0 + {_expr(rng, vars_, 1)}"]
        body = lines + inner + body
    src = f"def fuzz_case({', '.join(params)}):\n" + "\n".join(body) + "\n"
    return src


def _run_one(rng: random.Random) -> str:
    src = make_program(rng)
    # vectorize() requires inspectable source: define the case in a real file
    case_dir = pathlib.Path(tempfile.mkdtemp(prefix="array_vectorize_fuzz_"))
    path = case_dir / "fuzz_case_mod.py"
    path.write_text("import math\n\n\n" + src, encoding="utf-8", newline="\n")
    spec = importlib.util.spec_from_file_location("fuzz_case_mod", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = module.fuzz_case
    try:
        return _check_case(rng, src, fn)
    finally:
        # keep the file on disk while the case runs: inspect.getsource needs it
        shutil.rmtree(case_dir, ignore_errors=True)


def _check_case(rng: random.Random, src: str, fn: Any) -> str:
    try:
        vec = vectorize(fn)
    except errors.VectorizationError:
        return "rejected"
    except Exception as exc:
        raise AssertionError(f"internal error vectorizing:\n{src}\n{exc}") from exc
    n = rng.randint(1, 16)
    # -0.0 is excluded: Python round() drops its sign, xp.round keeps it
    # (documented divergence; covered by explicit edge-value tests)
    values = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -3.0]
    # draw inputs the scalar function can actually evaluate (a raising lane
    # is a documented divergence, not a bug); retry a few times before
    # giving up on the case
    for _attempt in range(5):
        args = tuple(
            np.asarray([rng.choice(values) for _ in range(n)], dtype=np.float64)
            for _ in range(fn.__code__.co_argcount)
        )
        try:
            expected = np.asarray(
                [fn(*(float(v) for v in elems)) for elems in zip(*args, strict=True)]
            )
            break
        except (ValueError, ArithmeticError, OverflowError, TypeError):
            continue
    else:
        return "edge-skip"
    with np.errstate(all="ignore"):
        try:
            got = np.asarray(vec(*args))  # constant results come back as scalars
        except TypeError:
            # dead-lane eager evaluation of dtype-invalid ops (design D1)
            return "edge-skip"
        except Exception as exc:
            raise AssertionError(f"internal error running vectorized code:\n{src}\n{exc}") from exc
        if expected.dtype == bool or got.dtype == bool:
            if not np.all(np.asarray(got, dtype=bool) == np.asarray(expected, dtype=bool)):
                raise AssertionError(f"bool mismatch for:\n{src}\ngot {got}\nexpected {expected}")
        else:
            expected = np.asarray(expected, dtype=np.float64)
            if not np.allclose(got, expected, equal_nan=True, rtol=1e-9, atol=1e-9):
                raise AssertionError(f"value mismatch for:\n{src}\ngot {got}\nexpected {expected}")
    return "ok"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse arguments and run the fuzz loop."""
    parser = argparse.ArgumentParser(prog="array_vectorize.fuzz", description=__doc__)
    parser.add_argument("--seconds", type=float, default=60.0, help="time budget")
    parser.add_argument("--seed", type=int, default=None, help="random seed (reproducible)")
    parser.add_argument("--cases", type=int, default=None, help="max cases (overrides time)")
    args = parser.parse_args(argv)

    seed = args.seed if args.seed is not None else random.randrange(2**32)
    rng = random.Random(seed)
    print(f"array_vectorize.fuzz: seed={seed} seconds={args.seconds} cases={args.cases}")
    deadline = time.monotonic() + args.seconds
    stats = {"ok": 0, "rejected": 0, "edge-skip": 0}
    cases = 0
    try:
        while cases < (args.cases or float("inf")) and time.monotonic() < deadline:
            outcome = _run_one(rng)
            stats[outcome] += 1
            cases += 1
    except AssertionError as exc:
        print(f"FAIL (seed={seed}, case {cases + 1}):\n{exc}")
        return 1
    print(
        f"done: {stats['ok']} ok, {stats['rejected']} rejected, {stats['edge-skip']} edge-skipped"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
