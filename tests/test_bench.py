"""Benchmark suite (pytest-benchmark, slow-marked).

Measures three things:

1. per-kernel speedups — vectorized output vs the scalar oracle
   (``np.vectorize``), across input sizes;
2. compile time — the ``vectorize()`` call itself, which matters for
   interactive use;
3. runtime-helper dispatch overhead — the exactness helpers must not
   regress plain float paths.

Run with ``make bench`` (``--benchmark-disable`` is the default so the
regular suite never pays the timing cost). Compare runs with
``--benchmark-save``/``--benchmark-compare`` or export with
``--benchmark-json``.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from vectorizer import vectorize
from vectorizer._runtime import _ARITH_OPS, _vec_arith, _vec_minmax

pytestmark = pytest.mark.slow

spec = importlib.util.spec_from_file_location("vec_corpus_b", Path(__file__).parent / "corpus.py")
assert spec is not None and spec.loader is not None
CORPUS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CORPUS)

# representative kernels: plain arithmetic, branch-heavy, loop-accumulated
# (name, function, positional argument count)
KERNELS = [
    ("add", CORPUS.add, 2),
    ("arith_ops", CORPUS.arith_ops, 2),
    ("relu", CORPUS.relu, 1),
    ("psi", CORPUS.psi, 1),
    ("clamp", CORPUS.clamp, 1),
]
SIZES = [1_000, 100_000, 1_000_000]

_add_id = _ARITH_OPS.index("add")


@pytest.mark.parametrize("name,kernel,nargs", KERNELS, ids=[k[0] for k in KERNELS])
@pytest.mark.parametrize("size", SIZES)
def test_kernel_vs_oracle(benchmark, name, kernel, nargs, size) -> None:
    """Vectorized kernel vs np.vectorize scalar oracle (per-element time)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(size)
    y = rng.standard_normal(size)
    args = (x, y) if nargs == 2 else (x,)
    small = tuple(a[:100] for a in args)
    vec = vectorize(kernel)
    oracle = np.vectorize(kernel)
    assert np.allclose(vec(*small), oracle(*small), equal_nan=True)

    benchmark(lambda: vec(*args))
    if not benchmark.disabled:
        per_elem_vec = benchmark.stats.stats.mean / size
        t_oracle = _best_of(lambda: oracle(*(a[:2_000] for a in args))) / 2_000
        print(
            f"{name}[{size}]: vectorized {t_oracle / per_elem_vec:.0f}x"
            " faster than the scalar oracle"
        )


def _best_of(fn, repeats: int = 3) -> float:
    import time

    best = math.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def test_compile_time(benchmark) -> None:
    """vectorize() on a branch + loop function (the expensive shape)."""

    def subject(x, y=2.0):
        # deliberately branch + loop shaped: this is the expensive
        # compile case (loop phis + where merges)
        acc = 0.0
        for _i in range(3):
            if x < 0:  # noqa: SIM108
                acc = acc + x * y
            else:
                acc = acc + math.sqrt(x)
        return acc

    vec = benchmark(vectorize, subject)
    assert vec(np.asarray([4.0]))[0] == pytest.approx(6.0)


def test_helper_dispatch_overhead(benchmark) -> None:
    """_vec_arith on plain floats must stay near the raw xp call."""
    xp = np
    a = xp.asarray([1.0] * 100_000)
    b = xp.asarray([2.0] * 100_000)

    def run() -> None:
        for _ in range(10):
            _vec_arith(xp, _add_id, a, b)

    benchmark(run)
    if not benchmark.disabled:
        base = _best_of(lambda: [a + b for _ in range(10)])
        stats = benchmark.stats.stats
        assert stats.mean <= 5 * base + 1e-4, (
            f"helper dispatch is {stats.mean / base:.1f}x the raw call"
        )


def test_minmax_dispatch_overhead(benchmark) -> None:
    """_vec_minmax on same-dtype floats must stay near xp.minimum."""
    xp = np
    a = xp.asarray([1.0] * 100_000)
    b = xp.asarray([2.0] * 100_000)

    def run() -> None:
        for _ in range(10):
            _vec_minmax(xp, False, a, b)

    benchmark(run)
    if not benchmark.disabled:
        base = _best_of(lambda: [xp.minimum(a, b) for _ in range(10)])
        stats = benchmark.stats.stats
        assert stats.mean <= 5 * base + 1e-4, (
            f"minmax helper is {stats.mean / base:.1f}x the raw call"
        )
