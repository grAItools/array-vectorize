"""Closures over scalars and arrays, keyword-only defaults."""

from __future__ import annotations

import corpus
import numpy as np
import support

import array_vectorize

X = np.asarray([-2.0, -0.5, 0.0, 0.5, 2.0, 10.0])


# ------------------------------------------------------- closures and defaults


def test_closure_scalar() -> None:
    assert np.allclose(support.vfn("closure_scalar")(X), X * corpus.SCALE)


def test_closure_array() -> None:
    got = support.vfn("closure_array")(np.asarray([1.0, 1.0, 1.0]))
    assert np.allclose(got, np.asarray([2.0, 3.0, 4.0]))


def test_kwonly_defaults() -> None:
    expected = X * 2.0 + 0.5
    assert np.allclose(support.vfn("kwonly_defaults")(X), expected)
    expected = X * 4.0 + 1.0
    assert np.allclose(support.vfn("kwonly_defaults")(X, scale=4.0, bias=1.0), expected)


def test_hidden_params_preserve_kwonly_defaults() -> None:
    mod = support.make_module(
        "import numpy as np\n"
        "C = np.asarray([2.0])\n"
        "\n"
        "\n"
        "def subject(x, *, scale=3):\n"
        "    return x * scale + C\n"
    )
    vec = array_vectorize.vectorize(mod.subject)
    assert np.allclose(vec(np.asarray([1.0])), [5.0])
    assert np.allclose(vec(np.asarray([1.0]), scale=4), [6.0])
