"""Optimizer passes: const-fold, CSE, DCE (plan §8) and the pass pipeline.

NOTE on imports: ``from vectorizer.optimize import cse`` (from-import) is
the supported form. The same-named package attributes hold the pass
functions after this package initializes, so attribute-style access like
``vectorizer.optimize.cse`` resolves to the function, not the submodule.
"""

from __future__ import annotations

from ..ir import Program
from ..ir.ssa import SSAEnv
from .constfold import const_fold
from .cse import cse
from .dce import dce
from .protect_domains import protect_domains

__all__ = ["const_fold", "cse", "dce", "optimize", "protect_domains"]


def optimize(program: Program, user_names: set[str] | None = None) -> Program:
    """Run const-fold, CSE, DCE (plan §8)."""
    folded = const_fold(program)
    ssa = SSAEnv(user_names or set())
    deduped = cse(folded, ssa)
    return dce(deduped)
