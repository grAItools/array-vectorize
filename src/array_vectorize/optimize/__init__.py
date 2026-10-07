"""Optimizer passes: const-fold, CSE, DCE and the pass pipeline.

NOTE on imports: ``from array_vectorize.optimize import cse`` (from-import) is
the supported form. The same-named package attributes hold the pass
functions after this package initializes, so attribute-style access like
``array_vectorize.optimize.cse`` resolves to the function, not the submodule.
"""

from __future__ import annotations

from array_vectorize.ir import Program
from array_vectorize.ir.ssa import SSAEnv
from array_vectorize.optimize.constfold import const_fold
from array_vectorize.optimize.cse import cse
from array_vectorize.optimize.dce import dce
from array_vectorize.optimize.protect_domains import protect_domains

__all__ = ["const_fold", "cse", "dce", "optimize", "protect_domains"]


def optimize(program: Program, user_names: set[str] | None = None) -> Program:
    """Run const-fold, CSE, DCE."""
    folded = const_fold(program)
    ssa = SSAEnv(user_names or set())
    deduped = cse(folded, ssa)
    return dce(deduped)
