"""Shared backend-detection helpers for verify and fallback.

``_is_array``/``_py_scalar`` are the two leaves both the differential
verifier and the element-loop fallback need; they live here so neither
duplicates them.
"""

from __future__ import annotations

from typing import Any

__all__ = ["_is_array", "_py_scalar"]


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
