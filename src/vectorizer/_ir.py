"""IR node dataclasses, the is_bool lattice, and SSA name allocation.

IR (per plan §6): frozen, hashable dataclasses. Hashability enables structural
CSE. ``DType`` is a small addition to the plan's node set, needed to represent
``xp.astype(x, xp.int64)`` targets.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


def generated_name(name: str) -> str:
    """The generated function's name; lambdas (``<lambda>``) are not identifiers."""
    return f"{name}_vec" if name.isidentifier() else "lambda_vec"


__all__ = [
    "BinOp",
    "Binding",
    "Call",
    "Compare",
    "DType",
    "FuncCall",
    "Literal",
    "Logical",
    "Loop",
    "Node",
    "Program",
    "Ref",
    "SSAEnv",
    "Stmt",
    "UnaryOp",
    "Where",
    "generated_name",
    "is_bool",
]

#: Names the generated code cannot rename: ``array_namespace`` is imported
#: by that name and ``hasattr`` is called by the namespace-detection
#: comprehension. User variables with these base names are mangled with a
#: trailing underscore. The namespace variable ``xp`` is NOT reserved: it is
#: chosen dynamically to avoid user names (see _lower.lower_function).
RESERVED_NAMES = frozenset({"array_namespace", "hasattr"})


def _literal_key(value: int | float | bool) -> tuple[Any, ...]:
    """Canonical key distinguishing -0.0 from 0.0 (copysign-observable)."""
    if isinstance(value, float) and value == 0.0:
        return (value, math.copysign(1.0, value))
    return (value,)


@dataclass(frozen=True, eq=False)
class Literal:
    """A scalar constant. ``inf``/``nan`` allowed (codegen emits xp.inf / xp.nan).

    Equality distinguishes signed zeros: ``0.0`` and ``-0.0`` compare unequal
    (functions like ``copysign`` observe the difference, so structural CSE
    must not merge them). NaN literals never compare equal.
    """

    value: int | float | bool
    kind: str  # 'int' | 'float' | 'bool'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Literal):
            return NotImplemented
        return self.kind == other.kind and _literal_key(self.value) == _literal_key(other.value)

    def __hash__(self) -> int:
        return hash((self.kind, self.value))


@dataclass(frozen=True)
class Ref:
    """Reference to a parameter or SSA binding by its emitted name."""

    name: str


@dataclass(frozen=True)
class BinOp:
    op: str  # add sub mul div pow floordiv mod and or xor lshift rshift
    left: Node
    right: Node


@dataclass(frozen=True)
class UnaryOp:
    op: str  # neg pos invert not
    operand: Node


@dataclass(frozen=True)
class Compare:
    op: str  # eq ne lt le gt ge
    left: Node
    right: Node


@dataclass(frozen=True)
class Logical:
    """``and``/``or`` over bool-typed operands only (numeric and/or lowers to Where)."""

    op: str  # 'and' | 'or'
    parts: tuple[Node, ...]


@dataclass(frozen=True)
class Where:
    cond: Node
    then: Node
    other: Node


@dataclass(frozen=True)
class Call:
    """Call to an Array API function, e.g. ``Call('sqrt', (x,))`` -> xp.sqrt(x)."""

    fn: str
    args: tuple[Node, ...]


@dataclass(frozen=True)
class DType:
    """An xp dtype reference; ``DType('int64')`` codegens to ``xp.int64``."""

    name: str


@dataclass(frozen=True)
class FuncCall:
    """Call to a vectorized helper function by its emitted name (plan D7)."""

    fn: str
    args: tuple[Node, ...]


Node = Literal | Ref | BinOp | UnaryOp | Compare | Logical | Where | Call | DType | FuncCall


@dataclass(frozen=True)
class Binding:
    """Let-statement: ``name = expr``."""

    name: str
    expr: Node


@dataclass(frozen=True)
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


Stmt = Binding | Loop


@dataclass(frozen=True)
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


class SSAEnv:
    """Allocates SSA names for parameters and bindings.

    Rules (plan §6):
    - first binding of a variable keeps its own name; rebinds pick ``x_N``
      with the smallest free N;
    - suffixes skip names used by the user (``x`` and ``x_1`` can both be
      user names);
    - user variables whose base name is reserved (``xp``) are mangled with a
      trailing underscore (``xp`` -> ``xp_``, ``xp_1``, ...).
    """

    def __init__(self, user_names: Iterable[str]) -> None:
        self._user = frozenset(user_names)
        self.emitted: set[str] = set()

    @property
    def user_names(self) -> frozenset[str]:
        return self._user

    def bind(self, var: str, *, force_suffix: bool = False) -> str:
        """Allocate the next binding name for ``var``.

        The first binding of a variable keeps its own name unless
        ``force_suffix`` (used inside branches, so the post-merge binding can
        take the clean base name). Reserved-base variables are mangled:
        ``xp`` -> ``xp_``, ``xp_1``, ... A mangled base may be another user
        variable's name, so it is only taken when no user name collides.
        """
        if var in RESERVED_NAMES:
            base = f"{var}_"
            if not force_suffix and base not in self.emitted and base not in self._user:
                self.emitted.add(base)
                return base
        elif not force_suffix and var not in self.emitted:
            self.emitted.add(var)
            return var
        i = 1
        while True:
            cand = f"{var}_{i}"
            if cand not in self.emitted and cand not in self._user:
                self.emitted.add(cand)
                return cand
            i += 1

    def fresh_temp(self, prefix: str) -> str:
        """Allocate a temp name ``<prefix>_N`` that no user name or binding uses."""
        i = 1
        while True:
            cand = f"{prefix}_{i}"
            if cand not in self.emitted and cand not in self._user:
                self.emitted.add(cand)
                return cand
            i += 1

    def reserve(self, name: str) -> None:
        """Mark ``name`` as taken (e.g. a generated helper function name)."""
        self.emitted.add(name)
        self._user = self._user | {name}

    def is_free(self, name: str) -> bool:
        """True if ``name`` is neither emitted nor used by the user."""
        return name not in self.emitted and name not in self._user
