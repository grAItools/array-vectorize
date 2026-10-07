"""FunctionInfo, Param — dependency-free types shared by the frontend stages."""

from __future__ import annotations

import ast
from collections.abc import Callable
import dataclasses
from typing import Any, Literal

__all__ = ["FunctionInfo", "Param", "ParamKind"]

_AstFunction = ast.FunctionDef | ast.Lambda

#: Parameter kinds, mirroring Python argument kinds.
type ParamKind = Literal["posonly", "arg", "kwonly"]


@dataclasses.dataclass(frozen=True, slots=True)
class Param:
    """One function parameter; ``kind`` mirrors Python argument kinds."""

    name: str
    kind: ParamKind
    default: int | float | bool | None = None
    has_default: bool = False


@dataclasses.dataclass(slots=True)
class FunctionInfo:
    """Everything the later passes need about the scalar function."""

    name: str
    filename: str
    source: str  # dedented original source text (embedded verbatim in the docstring)
    tree: _AstFunction
    params: list[Param]
    docstring: str | None
    closure_scalars: dict[str, int | float | bool] = dataclasses.field(default_factory=dict)
    closure_arrays: dict[str, Any] = dataclasses.field(default_factory=dict)
    math_funcs: dict[str, str] = dataclasses.field(default_factory=dict)  # name -> math attr
    math_modules: set[str] = dataclasses.field(default_factory=set)  # names bound to `math`
    user_funcs: dict[str, Callable[..., Any]] = dataclasses.field(default_factory=dict)
    user_names: set[str] = dataclasses.field(default_factory=set)

    @property
    def param_names(self) -> list[str]:
        """The parameter names, in signature order."""
        return [p.name for p in self.params]
