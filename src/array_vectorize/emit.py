"""GENERATION-TIME loading of vectorized functions (from _runtime).

``compile_vectorized`` runs once, inside ``vectorize()`` on the compiler
side: it compiles and execs the generated module source, registers it in
``linecache``, and attaches the wrapper attributes.

This is deliberately separate from ``array_vectorize.runtime``, the CALL-TIME
library (``_vec_arith``, ``_vec_minmax``, dtype analysis) injected into
generated modules and executed on the user's backend: generation-time
and call-time code never share a module.
"""

from __future__ import annotations

import inspect
import linecache
from collections.abc import Callable
from typing import Any, cast

from array_vectorize.ir import ssa

__all__ = ["compile_vectorized"]


def compile_vectorized(
    source: str,
    name: str,
    hidden_params: list[tuple[str, Any]],
    original: Callable[..., Any],
    helpers: list[tuple[str, Any]] | None = None,
) -> Callable[..., Any]:
    """Compile the generated module and return the callable itself.

    - compiles as ``<array_vectorize:{name}>`` and registers the source in
      ``linecache`` with ``mtime=None`` (survives ``checkcache``), so
      ``inspect.getsource`` and tracebacks show the real generated lines;
    - injects closure-array defaults into ``__kwdefaults__`` and vectorized
      helper functions into the module namespace;
    - sets ``.source`` and the ``_vectorized_original`` marker.
    """
    filename = f"<array_vectorize:{name}>"
    code = compile(source, filename, "exec")
    namespace: dict[str, Any] = {}
    exec(code, namespace)
    # retrieve the entry point BEFORE injecting helpers, so a helper whose
    # generated name collides can never shadow the function itself
    func: Any = namespace[ssa.generated_name(name)]
    # helper functions (vectorized user helpers AND the runtime dtype
    # promotion helpers) are injected here; bare-name calls resolve through
    # the function's globals at call time
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
    # module attribution: pydoc renders "Help on function f in module m"
    # instead of a moduleless header; safe because getsource/getsourcelines
    # use co_filename + linecache, never __module__
    func.__module__ = original.__module__
    # NOTE: __wrapped__ is intentionally NOT set: inspect.getsourcelines
    # unwraps unconditionally, which would hide the generated source. The
    # signature is preserved via __signature__ and the original is reachable
    # via the _vectorized_original marker (see _extract).
    func.__signature__ = inspect.signature(original)
    func._vectorized_original = original
    func.source = source
    return cast(Callable[..., Any], func)
