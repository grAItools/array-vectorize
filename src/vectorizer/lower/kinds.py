"""The explicit kind lattice: per-name kind-inference facts.

One :class:`VarInfo` per tracked (emitted) name consolidates the
per-name concerns; :class:`Kinds` adds the per-loop carried-frame
stack on top — together they replace the five parallel dicts the
lowerer previously smeared this state across:

- ``kind`` (was ``name_kinds``): the provable numeric kind — ``'int'`` /
  ``'float'`` / ``'bool'``, or ``None`` when unknown. Names are unique
  (SSA), so entries never go stale; this lets ``_numeric_kind`` see
  through ``Ref`` nodes.
- ``literal`` (was ``name_literals``): the :class:`~vectorizer.ir.Literal`
  value behind a binding of a plain literal, so later uses substitute the
  value instead of emitting casts on runtime plain scalars (which would
  crash ``xp.astype``/``xp.sqrt``).
- ``maybe_scalar`` (was ``_scalar_names``): the runtime value may be a raw
  Python scalar (parameter default, loop variable, literal binding, or a
  local computed from one). Operators promote scalars fine; ``xp.*``
  function arguments do not (strict backends reject plain scalars).
- ``maybe_bool`` (was ``_maybe_bool_names``): the value is a min/max
  result with unknown static kind whose runtime dtype may be boolean, so
  arithmetic must use the runtime-polymorphic intify.
- carried-assign kinds (was ``_carried_assign_kinds``): per active loop,
  the kinds assigned to loop-carried names in the body; a carried
  variable's runtime kind is the union of its phi kind and every body
  assignment.

Fixed-point snapshot/restore semantics: loop lowering re-lowers bodies
until per-name kinds stop widening. :meth:`Kinds.snapshot_facts` captures
kind + literal facts ONLY — never flags — and :meth:`Kinds.restore_facts`
replaces exactly those facts (names absent from the snapshot lose their
facts; post-snapshot facts vanish), reproducing the old wholesale dict
replacement. Flag and carried-frame state deliberately persists across
re-lowering passes.

Kind-fact *presence* is tracked separately from the ``kind`` value
(``_kind_established``): ``None`` is a legitimate kind value ("unknown"),
but a present-``None`` fact still blocks :meth:`Kinds.setdefault_kind`
exactly like a key in the old ``name_kinds`` dict did — and a fact dropped
by :meth:`Kinds.restore_facts` un-blocks it again.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..ir import Kind, Literal

__all__ = ["FactsSnapshot", "Kinds", "VarInfo"]


@dataclass
class VarInfo:
    """Per-name kind-inference facts (replaces the five parallel dicts)."""

    kind: Kind | None = None
    literal: Literal | None = None
    maybe_scalar: bool = False
    maybe_bool: bool = False


@dataclass(frozen=True)
class FactsSnapshot:
    """Presence-faithful copy of the kind + literal facts.

    ``kinds`` mirrors the old ``name_kinds`` dict including its key
    presence (a ``None`` value means "fact established, kind unknown");
    ``literals`` mirrors ``name_literals`` (presence == value is not None,
    so values are always real literals).
    """

    kinds: dict[str, Kind | None]
    literals: dict[str, Literal]


class Kinds:
    """Kind/literal/flag facts for every emitted name, plus the stack of
    per-loop carried-assignment kind frames."""

    def __init__(self) -> None:
        self._names: dict[str, VarInfo] = {}
        # names with an established kind fact (a None kind is a legitimate
        # "unknown" VALUE — presence is what setdefault consults)
        self._kind_established: set[str] = set()
        # per active loop: {carried name -> kinds assigned in the body}
        self.carried_frames: list[dict[str, set[Kind | None]]] = []

    def _entry(self, name: str) -> VarInfo:
        info = self._names.get(name)
        if info is None:
            info = self._names[name] = VarInfo()
        return info

    # ------------------------------------------------------------ kind facts

    def kind(self, name: str) -> Kind | None:
        """The name's recorded kind; absent names read as ``None``."""
        if name not in self._kind_established:
            return None
        return self._names[name].kind

    def set_kind(self, name: str, kind: Kind | None) -> None:
        self._kind_established.add(name)
        self._entry(name).kind = kind

    def setdefault_kind(self, name: str, kind: Kind | None) -> None:
        """Insert-only: an established fact (even ``kind=None``) wins."""
        if name not in self._kind_established:
            self.set_kind(name, kind)

    def update_kinds(self, labels: Mapping[str, Kind | None]) -> None:
        for name, kind in labels.items():
            self.set_kind(name, kind)

    # --------------------------------------------------------- literal facts

    def literal(self, name: str) -> Literal | None:
        info = self._names.get(name)
        return info.literal if info is not None else None

    def set_literal(self, name: str, lit: Literal) -> None:
        self._entry(name).literal = lit

    def drop_literal(self, name: str) -> None:
        info = self._names.get(name)
        if info is not None:
            info.literal = None

    # ----------------------------------------------------------------- flags

    def is_scalar(self, name: str) -> bool:
        info = self._names.get(name)
        return info is not None and (info.maybe_scalar or info.literal is not None)

    def mark_scalar(self, name: str) -> None:
        self._entry(name).maybe_scalar = True

    def unmark_scalar(self, name: str) -> None:
        info = self._names.get(name)
        if info is not None:
            info.maybe_scalar = False

    def is_maybe_bool(self, name: str) -> bool:
        info = self._names.get(name)
        return info is not None and info.maybe_bool

    def mark_maybe_bool(self, name: str) -> None:
        self._entry(name).maybe_bool = True

    def unmark_maybe_bool(self, name: str) -> None:
        info = self._names.get(name)
        if info is not None:
            info.maybe_bool = False

    # -------------------------------------------------------- carried frames

    def push_carried_frame(self) -> None:
        self.carried_frames.append({})

    def pop_carried_frame(self) -> dict[str, set[Kind | None]]:
        return self.carried_frames.pop()

    def record_carried_kind(self, name: str, kind: Kind | None) -> None:
        """Record a body-assignment kind for ``name`` in EVERY active frame."""
        for frame in self.carried_frames:
            frame.setdefault(name, set()).add(kind)

    # --------------------------------------- fixed-point snapshot and restore

    def snapshot_facts(self) -> FactsSnapshot:
        """Capture kind + literal facts ONLY — never flags."""
        kinds = {
            name: info.kind for name, info in self._names.items() if name in self._kind_established
        }
        literals = {
            name: info.literal for name, info in self._names.items() if info.literal is not None
        }
        return FactsSnapshot(kinds=kinds, literals=literals)

    def restore_facts(self, snapshot: FactsSnapshot) -> None:
        """REPLACE the kind + literal facts with ``snapshot``.

        True replacement semantics, like the old wholesale dict
        reassignment: names absent from the snapshot lose their facts
        (an established kind fact is revoked, so a later setdefault can
        fire again — the loop-var ``"int"`` insert on each fixed-point
        pass depends on this), and facts created after the snapshot
        vanish. Flags and carried frames are never touched.
        """
        self._kind_established = set(snapshot.kinds)
        for name, kind in snapshot.kinds.items():
            self._entry(name).kind = kind
        for name, lit in snapshot.literals.items():
            self._entry(name).literal = lit
        for name, info in self._names.items():
            if name not in snapshot.kinds:
                info.kind = None  # value irrelevant: presence was revoked
            if name not in snapshot.literals:
                info.literal = None
