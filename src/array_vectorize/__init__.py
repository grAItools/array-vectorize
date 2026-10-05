"""vectorize(f): compile a scalar Python function into an Array API vectorized function.

Public API re-export only — implementation lives in ``array_vectorize.api``
(and ``array_vectorize.pipeline`` for the strict compilation stages).
"""

from __future__ import annotations

from .api import get_source, vectorize
from .errors import VectorizationError

__version__ = "0.1.0"

__all__ = ["VectorizationError", "get_source", "vectorize"]
