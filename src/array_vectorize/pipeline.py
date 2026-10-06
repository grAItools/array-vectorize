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

from array_vectorize.codegen import generate_source
from array_vectorize.emit import compile_vectorized
from array_vectorize.frontend.extract import extract_function, resolve_original
from array_vectorize.frontend.validate import validate
from array_vectorize.lower import HelperVectorizer, lower_function
from array_vectorize.optimize import optimize, protect_domains
from array_vectorize.verify import verify_match

__all__ = ["compile_function"]


def compile_function(
    func: Callable[..., Any],
    *,
    protect: bool = False,
    verify_args: tuple[Any, ...] | None = None,
    helper_vectorizer: HelperVectorizer | None = None,
    namespace: Any = None,
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

    With ``namespace`` set (pinned mode), the generated code binds ``xp``
    to that namespace directly instead of extracting it from the arguments
    (so all-scalar calls become legal); the namespace is threaded to
    helper vectorization and verification.
    """
    info = extract_function(func)
    validate(info)
    lowered = lower_function(info, helper_vectorizer=helper_vectorizer)
    program = optimize(lowered.program, user_names=info.user_names | lowered.emitted_names)
    if protect:
        program = protect_domains(program)
    source = generate_source(lowered, program, pinned=namespace is not None)
    hidden_params = lowered.hidden_params
    if namespace is not None:
        # the pinned namespace rides as the hidden kw-only parameter's
        # runtime default (the existing __kwdefaults__ injection handles it)
        hidden_params = [*lowered.hidden_params, (lowered.namespace_param, namespace)]
    vec = compile_vectorized(
        source, lowered.name, hidden_params, original=func, helpers=lowered.helpers
    )
    if verify_args is not None:
        # compare against the resolved scalar original (what was actually
        # compiled), not the callable as passed — a prior vectorization
        # passed back in cannot serve as the element-wise oracle
        verify_match(vec, resolve_original(func), verify_args, namespace=namespace)
    return vec
