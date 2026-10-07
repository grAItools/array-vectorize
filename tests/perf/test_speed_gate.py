"""Performance smoke test (slow-marked).

Two gates:

1. Load-immune (primary): the vectorized function must be within a small
   factor of the hand-written array expression. Both sides are single numpy
   passes, so machine load slows them equally and the ratio stays stable.
   This is the actual claim: vectorization produced real array code.
2. Oracle comparison (secondary): >= 10x faster than ``np.vectorize``.
   This measures Python-call overhead elimination and is inherently
   sensitive to machine load; best-of-3 timing keeps it stable on quiet
   machines, but a heavily loaded machine can legitimately fail it.
"""

from __future__ import annotations

from collections.abc import Callable
import math
import time

import numpy as np
import pytest

import array_vectorize

pytestmark = pytest.mark.slow


def _best_of(fn: Callable[[], object], repeats: int = 3) -> float:
    """Best (minimum) wall time over ``repeats`` runs.

    The standard noise-robust timing statistic: scheduler hiccups and
    cache pollution only ever make a run slower.
    """
    best = math.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


@pytest.mark.timeout(120)
def test_faster_than_np_vectorize() -> None:
    pytest.importorskip("pytest_timeout")
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1_000_000)
    # a single-pass function maximizes the ratio: the comparison measures
    # the Python-call overhead that vectorization eliminates (multi-op or
    # exp-heavy bodies are Amdahl-limited by the array work itself).
    vec = array_vectorize.vectorize(lambda v: v + 1.0)
    oracle = np.vectorize(lambda v: v + 1.0)

    expected = oracle(x[:1000])  # warm up + sanity
    assert np.allclose(vec(x[:1000]), expected, equal_nan=True)

    t_vec = _best_of(lambda: vec(x))
    t_np = _best_of(lambda: x + 1.0)
    t_oracle = _best_of(lambda: oracle(x[:20_000])) * (1_000_000 / 20_000)

    # primary: within 5x of the hand-written expression (load-immune)
    assert t_vec <= 5 * t_np, (
        f"vectorized code is {t_vec / t_np:.1f}x slower than the hand-written expression"
    )
    # secondary: >= 10x faster than the scalar oracle (see module docstring
    # for the load-sensitivity caveat)
    ratio = t_oracle / max(t_vec, 1e-12)
    assert ratio >= 10, f"only {ratio:.1f}x faster than np.vectorize"
