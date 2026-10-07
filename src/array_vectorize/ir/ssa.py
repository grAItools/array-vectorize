# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""SSA name allocation (from _ir): generated names and the SSA environment."""

from __future__ import annotations

from collections.abc import Iterable

__all__ = ["RESERVED_NAMES", "SSAEnv", "generated_name"]


def generated_name(name: str) -> str:
    """The generated function's name; lambdas (``<lambda>``) are not identifiers."""
    return f"{name}_vec" if name.isidentifier() else "lambda_vec"


#: Names the generated code cannot rename: ``array_namespace`` is imported
#: by that name. ``hasattr`` is kept reserved defensively: generated code no
#: longer calls it (the namespace preamble passes arguments straight to
#: array_namespace), but reserving it stays zero-risk and keeps older
#: snapshots' name allocation stable. User variables with these base names
#: are mangled with a trailing underscore. The namespace variable ``xp``
#: is NOT reserved: it is chosen dynamically to avoid user names
#: (see _lower.lower_function).
RESERVED_NAMES = frozenset({"array_namespace", "hasattr"})


class SSAEnv:
    """Allocates SSA names for parameters and bindings.

    Rules:
    - first binding of a variable keeps its own name; rebinds pick ``x_N``
      with the smallest free N;
    - suffixes skip names used by the user (``x`` and ``x_1`` can both be
      user names);
    - user variables whose base name is reserved (``xp``) are mangled with a
      trailing underscore (``xp`` -> ``xp_``, ``xp_1``, ...).
    """

    def __init__(self, user_names: Iterable[str]) -> None:
        """Record the user names the allocator must avoid."""
        self._user = frozenset(user_names)
        self.emitted: set[str] = set()

    @property
    def user_names(self) -> frozenset[str]:
        """The user names this environment allocates around."""
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
