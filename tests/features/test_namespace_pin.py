# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""namespace= pinning and with_namespace variants (Strategy A).

A pinned vectorized function binds ``xp`` directly to the provided
namespace instead of extracting it from its arguments; all-scalar calls
become legal. ``with_namespace(xp)`` returns a NEW pinned callable.
"""

from __future__ import annotations

import inspect
import pathlib
import warnings

import corpus
import numpy as np
import pytest
import support

import array_vectorize
from array_vectorize import api

X = np.asarray([-2.0, -0.5, 0.0, 0.5, 2.0, 10.0])
Y = np.asarray([3.0, 1.5, 1.0, 0.5, -1.0, -3.0])


# ---- pinned source shape


def test_pinned_source_shape() -> None:
    vec = array_vectorize.vectorize(corpus.add, namespace=np)
    assert "def add_vec(x, y, *, _namespace=None):" in support.with_metadata(vec).source
    assert "xp = _namespace" in support.with_metadata(vec).source
    # pinned generated source has zero imports and no namespace extraction
    assert "import" not in support.with_metadata(vec).source
    assert "array_namespace" not in support.with_metadata(vec).source
    assert vec.__kwdefaults__ is not None
    assert vec.__kwdefaults__["_namespace"] is np


def test_unpinned_source_unchanged() -> None:
    unpinned = array_vectorize.vectorize(corpus.add)
    pinned = array_vectorize.vectorize(corpus.add, namespace=np)
    # Byte-identical to the golden source, excluding its repository header.
    golden = (pathlib.Path(__file__).parent.parent / "golden" / "cases" / "add.py").read_text(
        encoding="utf-8"
    )
    header = (pathlib.Path(__file__).parents[2] / ".license-header.txt").read_text(encoding="utf-8")
    assert golden.startswith(header + "\n")
    assert support.with_metadata(unpinned).source == golden.removeprefix(header + "\n")
    assert support.with_metadata(pinned).source != support.with_metadata(unpinned).source
    assert "xp = _namespace" not in support.with_metadata(unpinned).source
    assert "_namespace=None" not in support.with_metadata(unpinned).source


def test_pinned_closure_array_still_hidden_param() -> None:
    vec = array_vectorize.vectorize(corpus.closure_array, namespace=np)
    assert "*, ARR=None, _namespace=None" in support.with_metadata(vec).source
    assert vec.__kwdefaults__ is not None
    assert vec.__kwdefaults__["_namespace"] is np
    assert np.allclose(vec(np.asarray([0.0, 1.0, 2.0])), [1.0, 3.0, 5.0])


# ---- behavior parity


@pytest.mark.parametrize("name", ["add", "relu", "psi", "piecewise", "math_calls"])
def test_pinned_numpy_matches_unpinned(name: str) -> None:
    fn = getattr(corpus, name)
    args = (X, Y) if name == "add" else (X,)
    assert np.allclose(
        array_vectorize.vectorize(fn, namespace=np)(*args),
        array_vectorize.vectorize(fn)(*args),
        equal_nan=True,
    )


def test_pinned_strict_matches_unpinned() -> None:
    import array_api_strict as xps

    a = xps.asarray([1.0, -2.0, 3.0])
    unpinned = array_vectorize.vectorize(corpus.relu)
    pinned = array_vectorize.vectorize(corpus.relu, namespace=xps)
    assert list(map(float, pinned(a))) == list(map(float, unpinned(a)))


# ---- all-scalar calls (impossible unpinned)


def test_all_scalar_call_with_pin() -> None:
    fn = support.make_fn("    return math.sqrt(x * x)", defaults="x")
    vec = array_vectorize.vectorize(fn, namespace=np)
    assert float(vec(3.0)) == pytest.approx(3.0)


def test_all_scalar_minmax_with_pin() -> None:
    fn = support.make_fn("    return min(x, 3.0) + math.sqrt(x)", defaults="x")
    vec = array_vectorize.vectorize(fn, namespace=np)
    assert float(vec(4.0)) == pytest.approx(3.0 + 2.0)


def test_all_scalar_call_without_pin_raises() -> None:
    fn = support.make_fn("    return math.sqrt(x * x)", defaults="x")
    vec = array_vectorize.vectorize(fn)
    # message wording varies across array-api-compat versions: 1.10 says
    # "Unrecognized array input", newer releases "requires at least one
    # non-scalar array input" — the contract is the TypeError itself
    with pytest.raises(TypeError, match=r"at least one non-scalar|Unrecognized array input"):
        vec(3.0)


# ---- with_namespace


def test_with_namespace_equivalent_results() -> None:
    vf = array_vectorize.vectorize(corpus.psi)
    assert np.allclose(support.with_metadata(vf).with_namespace(np)(X), vf(X))


def test_with_namespace_identity_stable() -> None:
    vf = array_vectorize.vectorize(corpus.psi)
    assert support.with_metadata(vf).with_namespace(np) is support.with_metadata(vf).with_namespace(
        np
    )


def test_with_namespace_memoized_across_compilations() -> None:
    # the cache keys on the scalar original, not on the compiled wrapper:
    # distinct compilations (protected ones are never memoized) share pins
    vf1 = array_vectorize.vectorize(corpus.psi, protect_domains=True)
    vf2 = array_vectorize.vectorize(corpus.psi, protect_domains=True)
    assert vf1 is not vf2
    assert support.with_metadata(vf1).with_namespace(np) is support.with_metadata(
        vf2
    ).with_namespace(np)


def test_with_namespace_does_not_mutate() -> None:
    vf = array_vectorize.vectorize(corpus.psi)
    w = support.with_metadata(vf).with_namespace(np)
    assert w is not vf
    # the original stays unpinned: no hidden namespace parameter at all
    assert "_namespace=None" not in support.with_metadata(vf).source
    assert (vf.__kwdefaults__ or {}).get("_namespace") is None


def test_with_namespace_chaining_replaces_pin() -> None:
    import array_api_strict as xps

    vf = array_vectorize.vectorize(corpus.psi)
    a = xps.asarray([-2.0, 0.5, 3.0])
    chained = support.with_metadata(vf).with_namespace(np).with_namespace(xps)
    assert list(map(float, chained(a))) == list(map(float, vf(a)))


def test_with_namespace_preserves_contract() -> None:
    vf = array_vectorize.vectorize(corpus.psi)
    w = support.with_metadata(vf).with_namespace(np)
    assert w._vectorized_original is corpus.psi  # the true scalar original
    assert inspect.signature(w) == inspect.signature(corpus.psi)
    assert (
        w.source
        == support.with_metadata(array_vectorize.vectorize(corpus.psi, namespace=np)).source
    )
    assert w.__kwdefaults__ is not None
    assert w.__kwdefaults__["_namespace"] is np
    assert "psi_vec" in inspect.getsource(w)  # linecache registration intact


def test_with_namespace_does_not_rerun_verify(monkeypatch: pytest.MonkeyPatch) -> None:
    # pipeline calls verify.verify_match at call time, so patching the
    # verify module's attribute is what the compiled path sees
    import array_vectorize.verify as verify_mod

    calls: list[int] = []
    monkeypatch.setattr(verify_mod, "verify_match", lambda *a, **k: calls.append(1))
    fn = support.make_fn("    return x + 1.0", defaults="x")
    vec = array_vectorize.vectorize(fn, verify=(np.asarray([1.0]),))
    assert calls  # the initial decoration-time verify ran
    calls.clear()
    support.with_metadata(vec).with_namespace(np)
    assert not calls  # variants reuse verified body code, no re-verification


# ---- helper propagation


def test_pinned_helper_all_scalar_call() -> None:
    mod = support.make_module(
        "def helper_inner(y):\n"
        "    return y * y + 1.0\n"
        "\n"
        "\n"
        "def helper_outer(x):\n"
        "    return helper_inner(x) + helper_inner(x * 2.0)\n"
    )
    vec = array_vectorize.vectorize(mod.helper_outer, namespace=np)
    # only legal because the helper is pinned too (its own array_namespace
    # extraction would fail on scalars)
    assert float(vec(2.0)) == (2.0**2 + 1.0) + (4.0**2 + 1.0)


def test_pinned_and_unpinned_helpers_are_distinct() -> None:
    mod = support.make_module(
        "def helper_inner(y):\n"
        "    return y * y + 1.0\n"
        "\n"
        "\n"
        "def helper_outer(x):\n"
        "    return helper_inner(x) + helper_inner(x * 2.0)\n"
    )
    array_vectorize.vectorize(mod.helper_outer)
    array_vectorize.vectorize(mod.helper_outer, namespace=np)
    unpinned = api._HELPER_CACHE[(mod.helper_inner, False, None)]
    pinned = api._HELPER_CACHE[(mod.helper_inner, False, np)]
    assert unpinned is not pinned
    assert "xp = _namespace" in support.with_metadata(pinned).source
    assert "array_namespace" in support.with_metadata(unpinned).source


# ---- name collisions


def test_user_param_named_namespace_is_kept() -> None:
    fn = support.make_fn("    return x + _namespace", defaults="x, _namespace")
    vec = array_vectorize.vectorize(fn, namespace=np)
    # the user's parameter keeps its name; ours is mangled
    assert (
        "def subject_vec(x, _namespace, *, _namespace_1=None):" in support.with_metadata(vec).source
    )
    assert "xp = _namespace_1" in support.with_metadata(vec).source
    assert vec.__kwdefaults__ is not None
    assert vec.__kwdefaults__["_namespace_1"] is np
    assert np.allclose(vec(np.asarray([1.0]), np.asarray([5.0])), [6.0])
    # user keyword calls keep working
    assert np.allclose(vec(np.asarray([1.0]), _namespace=np.asarray([5.0])), [6.0])


def test_user_local_named_namespace_is_kept() -> None:
    fn = support.make_fn("    _namespace = x + 1\n    return _namespace * 2", defaults="x")
    vec = array_vectorize.vectorize(fn, namespace=np)
    assert "_namespace = x + 1" in support.with_metadata(vec).source
    assert "xp = _namespace_1" in support.with_metadata(vec).source
    assert np.allclose(vec(np.asarray([1.0])), [4.0])


# ---- fallback


def test_fallback_with_pin() -> None:
    fn = support.make_fn("    while x > 0:\n        x = x - 1\n    return x", defaults="x")
    with pytest.warns(UserWarning, match="falling back") as record:
        vec = array_vectorize.vectorize(fn, fallback=True, namespace=np)
    assert len(record) == 1  # warns once, at decoration time
    assert np.allclose(vec(np.asarray([5.0, -2.0])), [0.0, -2.0])


def test_fallback_with_namespace() -> None:
    import array_api_strict as xps

    fn = support.make_fn("    while x > 0:\n        x = x - 1\n    return x", defaults="x")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vec = array_vectorize.vectorize(fn, fallback=True)
    pinned = support.with_metadata(vec).with_namespace(xps)
    assert list(map(float, pinned(xps.asarray([3.0])))) == [0.0]


def test_fallback_pin_still_requires_array_arg() -> None:
    fn = support.make_fn("    while x > 0:\n        x = x - 1\n    return x", defaults="x")
    with pytest.warns(UserWarning, match="falling back"):
        vec = array_vectorize.vectorize(fn, fallback=True, namespace=np)
    with pytest.raises(TypeError, match="requires at least one"):
        vec(3.0)


# ---- invalid namespaces


def test_invalid_namespace_at_vectorize() -> None:
    fn = support.make_fn("    return x + 1.0", defaults="x")
    with pytest.raises(TypeError, match="not an Array API namespace"):
        array_vectorize.vectorize(fn, namespace=object())


def test_invalid_namespace_at_with_namespace() -> None:
    vf = array_vectorize.vectorize(support.make_fn("    return x + 1.0", defaults="x"))
    with pytest.raises(TypeError, match="not an Array API namespace"):
        support.with_metadata(vf).with_namespace(object())


# ---- verify= with a pin


def test_verify_with_pin_passes() -> None:
    fn = support.make_fn(
        "    if x < 0:\n        return 0.0\n    return x * math.exp(-x)", defaults="x"
    )
    xs = np.asarray([-1.0, 0.0, 2.0])
    vec = array_vectorize.vectorize(fn, verify=(xs,), namespace=np)
    assert np.allclose(vec(xs), np.where(xs < 0, 0.0, xs * np.exp(-xs)))


def test_verify_with_pin_strict_backend() -> None:
    import array_api_strict as xps

    fn = support.make_fn("    return x * 2.0", defaults="x")
    a = xps.asarray([1.0, 2.0])
    vec = array_vectorize.vectorize(fn, verify=(a,), namespace=xps)
    assert list(map(float, vec(a))) == [2.0, 4.0]
