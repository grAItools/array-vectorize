# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Shared backend-detection helpers for verify and fallback.

``_is_array``/``_py_scalar`` are the two leaves both the differential
verifier and the element-loop fallback need; ``_check_namespace`` validates
explicit namespace pins (``vectorize(namespace=...)``). They live here so
none of their users duplicates them (and so api.py and fallback.py can
share the check without an import cycle).

Array detection is fully delegated to ``array_api_compat``: no homegrown
``hasattr(value, "__array_namespace__")`` partial implementations of the
standard's detection machinery, which miss backends whose arrays never
exposed that attribute (torch.Tensor, CuPy).
"""

from __future__ import annotations

from typing import Any

import array_api_compat

__all__ = ["_check_namespace", "_is_array", "_py_scalar"]

#: members every Array API namespace is expected to expose
_NAMESPACE_MEMBERS = ("asarray", "where", "astype")


def _check_namespace(namespace: Any) -> None:
    """Validate an explicit namespace pin; raise ``TypeError`` if bogus.

    Passing something that is not an Array API namespace is a usage error
    (not a ``VectorizationError``): a valid namespace is a backend module
    such as ``numpy`` or ``array_api_strict``, or what
    ``array_namespace()`` returns for an array.
    """
    missing = [name for name in _NAMESPACE_MEMBERS if not callable(getattr(namespace, name, None))]
    if missing:
        raise TypeError(
            f"namespace {namespace!r} is not an Array API namespace "
            f"(missing: {', '.join(missing)}); pass a backend module such as "
            "numpy or array_api_strict, or what array_namespace() returns"
        )


def _is_array(value: Any) -> bool:
    """True if array-api-compat recognizes ``value`` as an array.

    Fully delegated to ``array_namespace``: it ignores Python scalars/None,
    dispatches torch tensors (which never exposed ``__array_namespace__``),
    and duck-types any standard-compliant array — no homegrown hasattr
    checks that miss backends.
    """
    try:
        array_api_compat.array_namespace(value)
    except TypeError:
        return False
    return True


def _py_scalar(value: Any) -> Any:
    """Backend element -> plain Python scalar (scalar code expects scalars).

    NumPy/Torch/JAX elements have ``.item()``; array-api-strict elements do
    not, so convert by dtype name instead. The oracle calls in verification
    and the element loop in fallback must run on plain Python scalars —
    backend scalars would change e.g. int/bool arithmetic semantics.
    """
    item = getattr(value, "item", None)
    if item is not None:
        return item()
    dtype = getattr(value, "dtype", None)
    # not backend detection: a conversion fallback for elements without
    # .item() (array-api-strict); torch/numpy/jax elements all have .item()
    if hasattr(value, "__array_namespace__") and dtype is not None:
        name = str(dtype)
        if "bool" in name:
            return bool(value)
        if "int" in name:
            return int(value)
        return float(value)
    return value
