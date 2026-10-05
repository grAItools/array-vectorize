"""Shared backend-detection helpers for verify and fallback.

``_is_array``/``_py_scalar`` are the two leaves both the differential
verifier and the element-loop fallback need; ``_check_namespace`` validates
explicit namespace pins (``vectorize(namespace=...)``). They live here so
none of their users duplicates them (and so api.py and fallback.py can
share the check without an import cycle).
"""

from __future__ import annotations

from typing import Any

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
    return hasattr(value, "__array_namespace__")


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
    if hasattr(value, "__array_namespace__") and dtype is not None:
        name = str(dtype)
        if "bool" in name:
            return bool(value)
        if "int" in name:
            return int(value)
        return float(value)
    return value
