"""jax.jit / torch.compile compatibility + backend-detection regression tests.

JAX and PyTorch trace and introspect the Python they compile — ``jax.jit``
runs the generated body under tracers (through the namespace-detection
preamble), ``torch.compile`` traces it with Dynamo — so the generated code
must be valid, inspectable, traceable Python. Every corpus function below
is exercised in four modes against its eager result:

- ``jax.jit`` with the namespace detected from the arguments
- ``jax.jit`` with the namespace pinned (``vectorize(fn, namespace=jax.numpy)``;
  the pinned generated source has no detection code at all)
- ``torch.compile`` with the namespace detected from the arguments
- ``torch.compile`` with the namespace pinned to ``array_api_compat.torch``
  (what ``array_namespace()`` returns for torch tensors)

Beyond compilation, this file regressions the backend detection delegated
to array-api-compat (``array_namespace``): ``verify=``, ``fallback=True``
and closure captures must recognize torch tensors, whose class has never
exposed ``__array_namespace__``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import importlib.util
import pathlib
from typing import Any

import numpy as np
import pytest
import support

import array_vectorize

spec = importlib.util.spec_from_file_location(
    "vec_corpus_c", pathlib.Path(__file__).parent.parent / "corpus.py"
)
assert spec is not None
assert spec.loader is not None
CORPUS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CORPUS)

X = [-2.0, -0.5, 0.0, 0.5, 2.0, 10.0]
Y = [3.0, 1.5, 1.0, 0.5, -1.0, -3.0]
# closure_array's captured ARR is [1.0, 2.0, 3.0]: a length-3 input
# broadcasts against it, and ARR is passed explicitly as a backend array —
# detected mode needs a single namespace among the arguments, and pinned
# torch must not hit NumPy's deprecated __array_wrap__ path.
X3 = [-2.0, 0.5, 3.0]
ARR = [1.0, 2.0, 3.0]

TWO_ARG = frozenset({"add", "math_two_arg", "minmax"})

NAMES = [
    "add",
    "relu",
    "relu_with_else",
    "psi",
    "piecewise",
    "nested_early_returns",
    "math_calls",
    "math_two_arg",
    "from_import",
    "closure_array",
    "closure_scalar",
    "loop_accumulate",
    "loop_start_stop_step",
    "loop_two_carried",
    "loop_with_branch",
    "nested_loops",
    "kwonly_defaults",
    "helper_outer",
    "loop_helper_caller",
    "ternary",
    "minmax",
    "cse_opportunity",
]


def _call_args(
    name: str, conv: Callable[[Sequence[float]], Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Positional args (plus the closure kwarg) for ``CORPUS.<name>``."""
    if name == "closure_array":
        return (conv(X3),), {"ARR": conv(ARR)}
    if name in TWO_ARG:
        return (conv(X), conv(Y)), {}
    return (conv(X),), {}


def _torch_conv(vals: Sequence[float]) -> Any:
    import torch

    return torch.tensor(vals, dtype=torch.float64)


# ---- jax.jit


@pytest.mark.parametrize("name", NAMES)
def test_jax_jit_detected_matches_eager(name: str) -> None:
    jax = pytest.importorskip("jax")
    # the generated math-call lowering casts through float64 (Python float
    # semantics); without x64 jax warns on every astype and truncates
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    fn = getattr(CORPUS, name)
    args, kwargs = _call_args(name, jnp.asarray)
    eager = array_vectorize.vectorize(fn)(*args, **kwargs)
    got = jax.jit(array_vectorize.vectorize(fn))(*args, **kwargs)
    assert np.allclose(np.asarray(got), np.asarray(eager), equal_nan=True)


@pytest.mark.parametrize("name", NAMES)
def test_jax_jit_pinned_matches_eager(name: str) -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    fn = getattr(CORPUS, name)
    args, kwargs = _call_args(name, jnp.asarray)
    eager = array_vectorize.vectorize(fn)(*args, **kwargs)
    got = jax.jit(array_vectorize.vectorize(fn, namespace=jnp))(*args, **kwargs)
    assert np.allclose(np.asarray(got), np.asarray(eager), equal_nan=True)


# ---- torch.compile
#
# NOTE: detected-mode torch used to fail with "TypeError: array_namespace
# requires at least one non-scalar array input" — the generated preamble
# filtered arguments with ``hasattr(a, '__array_namespace__')``, which
# torch.Tensor has never exposed (any 2.x). The preamble now passes the
# arguments straight to array_namespace() (array-api-compat >= 1.10 skips
# Python scalars/None and dispatches torch tensors), so all four modes
# below must pass.


@pytest.mark.timeout(600)
@pytest.mark.parametrize("name", NAMES)
def test_torch_compile_detected_matches_eager(name: str) -> None:
    pytest.importorskip("torch")
    import torch

    fn = getattr(CORPUS, name)
    args, kwargs = _call_args(name, _torch_conv)
    eager = array_vectorize.vectorize(fn)(*args, **kwargs)
    got = torch.compile(array_vectorize.vectorize(fn))(*args, **kwargs)
    assert torch.allclose(got, eager, equal_nan=True)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("name", NAMES)
def test_torch_compile_pinned_matches_eager(name: str) -> None:
    pytest.importorskip("torch")
    import array_api_compat.torch
    import torch

    fn = getattr(CORPUS, name)
    args, kwargs = _call_args(name, _torch_conv)
    eager = array_vectorize.vectorize(fn, namespace=array_api_compat.torch)(*args, **kwargs)
    got = torch.compile(array_vectorize.vectorize(fn, namespace=array_api_compat.torch))(
        *args, **kwargs
    )
    assert torch.allclose(got, eager, equal_nan=True)


# ---- backend-detection regression tests (delegated to array_namespace)
#
# verify=, fallback=True and closure captures used to detect arrays with
# a homegrown ``hasattr(value, '__array_namespace__')`` check, which
# torch.Tensor has never satisfied — these paths now delegate to
# array_api_compat's array_namespace and must work on torch tensors.


def test_verify_with_torch_args() -> None:
    pytest.importorskip("torch")
    import torch

    t = torch.tensor(X, dtype=torch.float64)
    # previously: VectorizationError("verify= needs at least one Array API
    # array in the example inputs")
    vec = array_vectorize.vectorize(CORPUS.psi, verify=(t,))
    got = vec(t)
    expected = torch.where(t < 0, torch.tensor(0.0), t * torch.exp(-t))
    assert torch.allclose(got, expected)


def test_fallback_with_torch_tensors() -> None:
    pytest.importorskip("torch")
    import torch

    fn = support.make_fn("    while x > 0:\n        x = x - 1\n    return x", defaults="x")
    with pytest.warns(UserWarning, match="falling back"):
        vec = array_vectorize.vectorize(fn, fallback=True)
    # previously: TypeError("vectorized fallback ... requires at least one
    # Array API array argument")
    t = torch.tensor([5.0, -2.0], dtype=torch.float64)
    got = vec(t)
    assert isinstance(got, torch.Tensor)
    # the element loop re-wraps Python scalars via xp.asarray, which on
    # torch lands on the default float32 — values, not dtype, are the
    # fallback contract (mirrors the numpy fallback tests' list compare)
    assert torch.allclose(got.double(), torch.tensor([0.0, -2.0], dtype=torch.float64))


def test_fallback_pinned_with_torch_tensors() -> None:
    pytest.importorskip("torch")
    import array_api_compat.torch
    import torch

    fn = support.make_fn("    while x > 0:\n        x = x - 1\n    return x", defaults="x")
    with pytest.warns(UserWarning, match="falling back"):
        vec = array_vectorize.vectorize(fn, fallback=True, namespace=array_api_compat.torch)
    t = torch.tensor([5.0, -2.0], dtype=torch.float64)
    got = vec(t)
    assert isinstance(got, torch.Tensor)
    assert torch.allclose(got.double(), torch.tensor([0.0, -2.0], dtype=torch.float64))


def test_closure_captured_torch_tensor() -> None:
    pytest.importorskip("torch")
    import torch

    mod = support.make_module(
        "import torch\n"
        "T = torch.asarray([1.0, 2.0, 3.0], dtype=torch.float64)\n"
        "\n"
        "\n"
        "def subject(x):\n"
        "    return x + T\n"
    )
    # previously: VectorizationError (closure/global 'T' has unsupported
    # type 'Tensor'); now a hidden kw-only closure param like numpy captures
    vec = array_vectorize.vectorize(mod.subject)
    assert "def subject_vec(x, *, T=None):" in vec.source
    assert vec.__kwdefaults__["T"] is mod.T
    t = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
    got = vec(t)
    assert torch.allclose(got, torch.tensor([11.0, 22.0, 33.0], dtype=torch.float64))


def test_closure_captured_jax_array() -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    mod = support.make_module(
        "import jax.numpy as jnp\n"
        "T = jnp.asarray([1.0, 2.0, 3.0])\n"
        "\n"
        "\n"
        "def subject(x):\n"
        "    return x + T\n"
    )
    vec = array_vectorize.vectorize(mod.subject)
    assert "def subject_vec(x, *, T=None):" in vec.source
    assert vec.__kwdefaults__["T"] is mod.T
    got = vec(mod.T * 10.0)
    assert np.allclose(np.asarray(got), [11.0, 22.0, 33.0])


@pytest.mark.timeout(600)
@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_compiled_integer_division_preserves_python_rounding(backend: str) -> None:
    scalar = support.make_fn("    return x / y", defaults="x, y")
    left = [1, -1, -(2**63), 2**63 - 1, 0]
    right = [2**53 + 1, 2**53 + 1, -1, 3, -1]
    if backend == "jax":
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        vec = array_vectorize.vectorize(scalar, namespace=jnp)
        got = jax.jit(vec)(jnp.asarray(left, dtype=jnp.int64), jnp.asarray(right, dtype=jnp.int64))
    else:
        torch = pytest.importorskip("torch")
        import array_api_compat.torch

        vec = array_vectorize.vectorize(scalar, namespace=array_api_compat.torch)
        got = torch.compile(vec)(
            torch.tensor(left, dtype=torch.int64), torch.tensor(right, dtype=torch.int64)
        )
    expected = [a / b for a, b in zip(left, right, strict=True)]
    np.testing.assert_array_equal(np.asarray(got), expected)
    np.testing.assert_array_equal(np.signbit(np.asarray(got)), np.signbit(expected))
