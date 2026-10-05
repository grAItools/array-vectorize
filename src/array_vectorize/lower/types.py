"""Lowering output types shared across the pipeline.

``LoweredFunction`` is the contract between lowering and everything
downstream (optimize, codegen, emit): the IR program plus the metadata
those stages need. ``HelperVectorizer`` is the callback type lowering
uses to vectorize user helper functions on demand.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..frontend.info import Param
from ..ir import Program

__all__ = ["HelperVectorizer", "LoweredFunction"]


@dataclass(slots=True)
class LoweredFunction:
    """Lowering output: IR plus everything codegen/runtime need."""

    program: Program
    name: str
    params: list[Param]
    param_names: list[str]  # emitted names, aligned with params
    hidden_params: list[tuple[str, Any]]  # (emitted name, array default)
    source: str  # original scalar source (embedded verbatim in the docstring)
    helpers: list[tuple[str, Any]]  # (emitted name, vectorized helper callable)
    namespace_var: str  # the generated code's ``xp`` (renamed on collision)
    #: the allocated, collision-free name of the hidden kw-only namespace
    #: parameter bound in pinned mode (``vectorize(namespace=...)``); never
    #: referenced by body IR, so it is not a program parameter
    namespace_param: str
    emitted_names: frozenset[str]  # every binding/param name the lowering emitted


type HelperVectorizer = Callable[[Callable[..., Any]], Callable[..., Any]]
