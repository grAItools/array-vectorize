"""vectorize(f): compile a scalar Python function into an Array API vectorized function."""

from __future__ import annotations

from typing import Any, Callable

from ._errors import VectorizationError
from ._extract import extract_function
from ._lower import lower_function
from ._optimize import optimize
from ._codegen import generate_source
from ._runtime import compile_vectorized
from ._validate import validate

__version__ = "0.1.0"

__all__ = ["vectorize", "VectorizationError", "get_source"]


def vectorize(
    func: Callable[..., Any],
    *,
    strict: bool = True,
    fallback: bool = False,
) -> Callable[..., Any]:
    """Compile an inspectable scalar function into a vectorized one.

    The returned callable runs over any Array API backend (NumPy, PyTorch,
    JAX, CuPy, array-api-strict, ...) using only standard functions. The
    generated source is available as ``.source`` and via ``inspect.getsource``.

    Raises :class:`VectorizationError` (never silently miscompiles) when the
    function uses constructs outside the supported subset.
    """
    if not strict or fallback:
        raise NotImplementedError("strict=False / fallback=True arrive in milestone M3")
    info = extract_function(func)
    validate(info)
    lowered = lower_function(info)
    program = optimize(lowered.program, user_names=info.user_names)
    source = generate_source(lowered, program)
    return compile_vectorized(source, lowered.name, lowered.hidden_params, original=func)


def get_source(vectorized: Callable[..., Any]) -> str:
    """Return the generated source of a vectorized function."""
    source = getattr(vectorized, "source", None)
    if not isinstance(source, str):
        raise VectorizationError(f"{vectorized!r} is not a vectorized function")
    return source
