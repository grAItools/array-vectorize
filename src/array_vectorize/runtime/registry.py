"""The runtime-helper registry: the seam between lowering and emitted calls.

Lowering decides WHERE a runtime helper is needed; this registry is the
single place that maps that decision to WHAT gets emitted: the helper's
base name in generated source and the callable injected into the
generated module's globals by emit.compile_vectorized.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from array_vectorize.runtime import arith, minmax

# lowering goes through the registry seam (docs/architecture.md):
# lower/expressions.py imports ArithOp from here, not from runtime.arith
from array_vectorize.runtime.arith import ArithOp  # cleanporter: ignore[CP003] re-export

__all__ = ["RUNTIME_HELPERS", "ArithOp"]

#: stable key -> (default emitted base name, injected callable).
#: The base names are load-bearing: generated source (and the golden
#: snapshots) reference them verbatim.
RUNTIME_HELPERS: dict[str, tuple[str, Callable[..., Any]]] = {
    "vec_arith": ("_vec_arith", arith._vec_arith),
    "vec_minmax": ("_vec_minmax", minmax._vec_minmax),
}
