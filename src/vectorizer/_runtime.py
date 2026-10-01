"""Compile/exec/linecache plumbing and wrapper attributes (plan §9)."""

from __future__ import annotations

import linecache
from typing import Any, Callable

__all__ = ["compile_vectorized"]


def compile_vectorized(
    source: str,
    name: str,
    hidden_params: list[tuple[str, Any]],
    original: Callable[..., Any],
) -> Callable[..., Any]:
    """Compile the generated module and return the callable itself.

    - compiles as ``<vectorizer:{name}>`` and registers the source in
      ``linecache`` with ``mtime=None`` (survives ``checkcache``), so
      ``inspect.getsource`` and tracebacks show the real generated lines;
    - injects closure-array defaults into ``__kwdefaults__``;
    - sets ``.source`` and ``__wrapped__`` (the original scalar function).
    """
    filename = f"<vectorizer:{name}>"
    code = compile(source, filename, "exec")  # noqa: S102
    namespace: dict[str, Any] = {}
    exec(code, namespace)  # noqa: S102
    func = namespace[f"{name}_vec"]

    if hidden_params:
        func.__kwdefaults__ = {param: value for param, value in hidden_params}

    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)

    func.__name__ = original.__name__
    func.__qualname__ = getattr(original, "__qualname__", original.__name__)
    func.__wrapped__ = original
    func.source = source  # type: ignore[attr-defined]
    return func
