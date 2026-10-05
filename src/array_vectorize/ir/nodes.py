"""IR node dataclasses and the is_bool lattice.

IR (per plan §6): frozen, hashable dataclasses. Hashability enables structural
CSE. ``DType`` is a small addition to the plan's node set, needed to represent
``xp.astype(x, xp.int64)`` targets.
"""

from __future__ import annotations

import math
import typing
from dataclasses import dataclass
from typing import Any

__all__ = [
    "BinOp",
    "Binding",
    "Call",
    "Compare",
    "DType",
    "FuncCall",
    "Kind",
    "Literal",
    "Logical",
    "Loop",
    "Node",
    "Program",
    "Ref",
    "Stmt",
    "UnaryOp",
    "Where",
    "is_bool",
]


def _literal_key(value: int | float | bool) -> tuple[Any, ...]:
    """Canonical key distinguishing -0.0 from 0.0 (copysign-observable)."""
    if isinstance(value, float) and value == 0.0:
        return (value, math.copysign(1.0, value))
    return (value,)


#: The numeric-kind lattice values tracked through lowering: the provable
#: kind of a literal, binding, or expression node.
type Kind = typing.Literal["int", "float", "bool"]


@dataclass(frozen=True, eq=False, slots=True)
class Literal:
    """A scalar constant. ``inf``/``nan`` allowed (codegen emits xp.inf / xp.nan).

    Equality distinguishes signed zeros: ``0.0`` and ``-0.0`` compare unequal
    (functions like ``copysign`` observe the difference, so structural CSE
    must not merge them). NaN literals never compare equal.
    """

    value: int | float | bool
    kind: Kind

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Literal):
            return NotImplemented
        return self.kind == other.kind and _literal_key(self.value) == _literal_key(other.value)

    def __hash__(self) -> int:
        return hash((self.kind, self.value))


@dataclass(frozen=True, slots=True)
class Ref:
    """Reference to a parameter or SSA binding by its emitted name."""

    name: str


@dataclass(frozen=True, slots=True)
class BinOp:
    op: str  # add sub mul div pow floordiv mod and or xor lshift rshift
    left: Node
    right: Node


@dataclass(frozen=True, slots=True)
class UnaryOp:
    op: str  # neg pos invert not
    operand: Node


@dataclass(frozen=True, slots=True)
class Compare:
    op: str  # eq ne lt le gt ge
    left: Node
    right: Node


@dataclass(frozen=True, slots=True)
class Logical:
    """``and``/``or`` over bool-typed operands only (numeric and/or lowers to Where)."""

    op: str  # 'and' | 'or'
    parts: tuple[Node, ...]


@dataclass(frozen=True, slots=True)
class Where:
    cond: Node
    then: Node
    other: Node


@dataclass(frozen=True, slots=True)
class Call:
    """Call to an Array API function, e.g. ``Call('sqrt', (x,))`` -> xp.sqrt(x)."""

    fn: str
    args: tuple[Node, ...]


@dataclass(frozen=True, slots=True)
class DType:
    """An xp dtype reference; ``DType('int64')`` codegens to ``xp.int64``."""

    name: str


@dataclass(frozen=True, slots=True)
class FuncCall:
    """Call to a vectorized helper function by its emitted name (plan D7)."""

    fn: str
    args: tuple[Node, ...]


type Node = Literal | Ref | BinOp | UnaryOp | Compare | Logical | Where | Call | DType | FuncCall


@dataclass(frozen=True, slots=True)
class Binding:
    """Let-statement: ``name = expr``."""

    name: str
    expr: Node


@dataclass(frozen=True, slots=True)
class Loop:
    """Constant-trip ``for`` loop emitted as a real loop (plan D6).

    ``start``/``stop``/``step`` are const int nodes (step ``Literal(1)`` when
    absent). Loop-carried variables are handled with explicit phi bindings
    emitted just before the loop (see _lower).
    """

    var: str
    start: Node
    stop: Node
    step: Node
    body: tuple[Stmt, ...]


type Stmt = Binding | Loop


@dataclass(frozen=True, slots=True)
class Program:
    params: tuple[str, ...]
    bindings: tuple[Stmt, ...]
    result: Node


def is_bool(node: Node) -> bool:
    """Conservative boolean lattice (plan §6): True only for provably-boolean nodes."""
    if isinstance(node, Compare):
        return True
    if isinstance(node, UnaryOp):
        return node.op == "not"
    if isinstance(node, Logical):
        return True
    if isinstance(node, Literal):
        return node.kind == "bool"
    if isinstance(node, Call):
        return node.fn in ("isnan", "isinf", "isfinite")
    return False
