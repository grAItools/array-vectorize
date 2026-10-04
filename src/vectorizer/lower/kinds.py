"""The explicit kind lattice for tracked lowering variables.

Planned consolidation (restructure §9 phase 4b): a single ``VarInfo``
dataclass per tracked variable — ``kind`` (``'int'`` / ``'float'`` /
``'bool'`` or ``None``), ``literal``, ``scalar``, ``maybe_bool`` —
replaces the five parallel tracking dicts the lowerer carries today
(``name_kinds``, ``name_literals``, ``_scalar_names``,
``_maybe_bool_names``, ``_carried_assign_kinds``), migrated one dict at
a time. Until that migration lands, this module re-exports the IR's
``Kind`` alias so the package shape matches the target design.
"""

from __future__ import annotations

from ..ir import Kind

__all__ = ["Kind"]
