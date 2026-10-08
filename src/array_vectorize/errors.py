# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Error types and diagnostics for vectorization failures."""

from __future__ import annotations

import dataclasses
from typing import NoReturn


@dataclasses.dataclass(frozen=True, slots=True)
class Diagnostic:
    """One unsupported-construct finding with its source location."""

    message: str
    lineno: int
    col_offset: int
    line: str

    def render(self) -> str:
        """Render the finding as a located snippet with a carets pointer."""
        pointer = " " * self.col_offset + "^"
        return (
            f"  line {self.lineno}, col {self.col_offset}: {self.message}\n"
            f"      {self.line.rstrip()}\n"
            f"      {pointer}"
        )


class VectorizationError(Exception):
    """Raised when a function cannot be vectorized.

    The validator collects *all* violations at once (linter-style); they are
    carried in ``diagnostics`` and rendered in the final message.
    """

    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None) -> None:
        """Keep the headline message and the collected diagnostics."""
        super().__init__(message)
        self.message = message
        self.diagnostics: list[Diagnostic] = list(diagnostics or [])

    def __str__(self) -> str:
        """The message, plus every rendered diagnostic below it."""
        if not self.diagnostics:
            return self.message
        rendered = "\n".join(d.render() for d in self.diagnostics)
        return f"{self.message}\n{rendered}"


def _reject(name: str, message: str) -> NoReturn:
    raise VectorizationError(f"cannot vectorize {name!r}: {message}")
