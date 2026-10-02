"""Compile/exec/linecache plumbing and wrapper attributes (plan §9)."""

from __future__ import annotations

import inspect
import linecache
from collections.abc import Callable
from typing import Any, cast

from ._ir import generated_name

__all__ = ["compile_vectorized"]


def compile_vectorized(
    source: str,
    name: str,
    hidden_params: list[tuple[str, Any]],
    original: Callable[..., Any],
    helpers: list[tuple[str, Any]] | None = None,
) -> Callable[..., Any]:
    """Compile the generated module and return the callable itself.

    - compiles as ``<vectorizer:{name}>`` and registers the source in
      ``linecache`` with ``mtime=None`` (survives ``checkcache``), so
      ``inspect.getsource`` and tracebacks show the real generated lines;
    - injects closure-array defaults into ``__kwdefaults__`` and vectorized
      helper functions into the module namespace;
    - sets ``.source`` and the ``_vectorized_original`` marker.
    """
    filename = f"<vectorizer:{name}>"
    code = compile(source, filename, "exec")
    namespace: dict[str, Any] = {}
    exec(code, namespace)
    # retrieve the entry point BEFORE injecting helpers, so a helper whose
    # generated name collides can never shadow the function itself
    func: Any = namespace[generated_name(name)]
    for helper_name, helper_fn in helpers or []:
        namespace[helper_name] = helper_fn

    if hidden_params:
        # merge: replacing __kwdefaults__ would erase the function's own
        # keyword-only defaults
        merged = {**getattr(func, "__kwdefaults__", {})}
        merged.update(dict(hidden_params))
        func.__kwdefaults__ = merged

    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)

    func.__name__ = original.__name__
    func.__qualname__ = getattr(original, "__qualname__", original.__name__)
    # NOTE: __wrapped__ is intentionally NOT set: inspect.getsourcelines
    # unwraps unconditionally, which would hide the generated source. The
    # signature is preserved via __signature__ and the original is reachable
    # via the _vectorized_original marker (see _extract).
    func.__signature__ = inspect.signature(original)
    func._vectorized_original = original
    func.source = source
    return cast(Callable[..., Any], func)
