"""The vectorization pipeline: extract -> validate -> lower -> optimize -> codegen -> compile.

This module owns the strict (non-fallback) compilation stages only; option
handling, the helper cache, and the fallback decision live in ``api.py``.
The pipeline never imports ``api`` — the ``helper_vectorizer`` callback is
injected by the caller, which is how helper recursion detection and caching
stay at the API edge without an import cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .codegen import generate_source
from .emit import compile_vectorized
from .frontend.extract import extract_function
from .frontend.validate import validate
from .lower import HelperVectorizer, lower_function
from .optimize import optimize, protect_domains
from .verify import verify_match

__all__ = ["compile_function"]


def compile_function(
    func: Callable[..., Any],
    *,
    protect: bool = False,
    verify_args: tuple[Any, ...] | None = None,
    helper_vectorizer: HelperVectorizer | None = None,
) -> Callable[..., Any]:
    """Run the strict pipeline over ``func``.

    Stages:

    1. **extract** (``frontend.extract``) — pull the function's source and
       closure captures into a ``FunctionInfo``.
    2. **validate** (``frontend.validate``) — check the supported-subset
       rules, collecting all diagnostics at once.
    3. **lower** (``lower``) — AST to IR, calling ``helper_vectorizer``
       (supplied by ``api``) for user-defined helpers.
    4. **optimize** (``optimize``) — const-fold, CSE, DCE; then the
       ``protect_domains`` post-pass when ``protect`` is set.
    5. **codegen** (``codegen``) — IR to readable Python source.
    6. **compile** (``emit``) — exec + linecache registration.
    7. **verify** (``verify``) — optional differential check against the
       scalar original on ``verify_args``.
    """
    info = extract_function(func)
    validate(info)
    lowered = lower_function(info, helper_vectorizer=helper_vectorizer)
    program = optimize(lowered.program, user_names=info.user_names | lowered.emitted_names)
    if protect:
        program = protect_domains(program)
    source = generate_source(lowered, program)
    vec = compile_vectorized(
        source, lowered.name, lowered.hidden_params, original=func, helpers=lowered.helpers
    )
    if verify_args is not None:
        verify_match(vec, func, verify_args)
    return vec
