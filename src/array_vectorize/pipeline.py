# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

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

from array_vectorize import codegen
from array_vectorize import emit
from array_vectorize import lower
from array_vectorize import optimize as optimize_mod
from array_vectorize import verify
from array_vectorize.frontend import extract
from array_vectorize.frontend import validate as validate_mod

# optimize's documented submodule/function name collision (the NOTE in
# optimize/__init__.py): the from-import is the supported spelling
from array_vectorize.optimize import protect_domains  # cleanporter: ignore[CP002] name collision

__all__ = ["compile_function"]


def compile_function(
    func: Callable[..., Any],
    *,
    protect: bool = False,
    verify_args: tuple[Any, ...] | None = None,
    helper_vectorizer: lower.HelperVectorizer | None = None,
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
    info = extract.extract_function(func)
    validate_mod.validate(info)
    lowered = lower.lower_function(info, helper_vectorizer=helper_vectorizer)
    program = optimize_mod.optimize(
        lowered.program, user_names=info.user_names | lowered.emitted_names
    )
    if protect:
        program = protect_domains(program)
    source = codegen.generate_source(lowered, program, pinned=namespace is not None)
    hidden_params = lowered.hidden_params
    if namespace is not None:
        # the pinned namespace rides as the hidden kw-only parameter's
        # runtime default (the existing __kwdefaults__ injection handles it)
        hidden_params = [*lowered.hidden_params, (lowered.namespace_param, namespace)]
    vec = emit.compile_vectorized(
        source, lowered.name, hidden_params, original=func, helpers=lowered.helpers
    )
    if verify_args is not None:
        # compare against the resolved scalar original (what was actually
        # compiled), not the callable as passed — a prior vectorization
        # passed back in cannot serve as the element-wise oracle
        verify.verify_match(vec, extract.resolve_original(func), verify_args, namespace=namespace)
    return vec
