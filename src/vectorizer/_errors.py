"""Error types and diagnostics for vectorization failures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """One unsupported-construct finding with its source location."""

    message: str
    lineno: int
    col_offset: int
    line: str

    def render(self) -> str:
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
        super().__init__(message)
        self.message = message
        self.diagnostics: list[Diagnostic] = list(diagnostics or [])

    def __str__(self) -> str:
        if not self.diagnostics:
            return self.message
        rendered = "\n".join(d.render() for d in self.diagnostics)
        return f"{self.message}\n{rendered}"
