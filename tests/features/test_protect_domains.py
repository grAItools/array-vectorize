"""protect_domains=True dead-lane clamping semantics."""

from __future__ import annotations

import math

import numpy as np
import support

import array_vectorize

# ---------------------------------------------------------- dead-lane clamping


def test_protect_domains_clamps_dead_lanes() -> None:
    fn = support.make_fn("    if x > 0:\n        return math.sqrt(x)\n    return 0.0")
    vec = array_vectorize.vectorize(fn, protect_domains=True)
    assert "xp.logical_not(x > 0)" in vec.source  # dead-lane guard
    xs = np.asarray([-1.0, 0.5, 4.0])
    with np.errstate(invalid="ignore", divide="ignore"):
        got = vec(xs)
    assert np.allclose(got, [0.0, np.sqrt(0.5), 2.0])


def test_protect_domains_no_warning() -> None:
    fn = support.make_fn("    if x > 0:\n        return math.sqrt(x)\n    return 0.0")
    vec = array_vectorize.vectorize(fn, protect_domains=True)
    xs = np.asarray([-100.0, -1.0])
    with np.errstate(all="raise"):  # any warning becomes an error
        got = vec(xs)
    assert np.allclose(got, [0.0, 0.0])


def test_protect_domains_no_warning_on_poles() -> None:
    vec = array_vectorize.vectorize(
        support.make_fn("    return math.log(x) if x > 0 else 0.0"), protect_domains=True
    )
    with np.errstate(all="raise"):
        got = vec(np.asarray([-1.0, 1.0]))
    assert np.allclose(got, [0.0, 0.0])


def test_protect_domains_open_boundaries_never_warn() -> None:
    vec = array_vectorize.vectorize(
        support.make_fn("    return math.log1p(x) if x > -1 else 0.0"), protect_domains=True
    )
    with np.errstate(all="raise"):
        got = vec(np.asarray([-1.0, 0.0]))
    assert np.allclose(got, [0.0, 0.0])


def test_protect_domains_two_sided_clamp() -> None:
    fn = support.make_fn("    if x != 0:\n        return math.atanh(x)\n    return 0.0")
    vec = array_vectorize.vectorize(fn, protect_domains=True)
    # dead-lane where-clamps (preserve -0.0 and live values, unlike maximum)
    assert "xp.logical_not" in vec.source
    assert "xp.maximum" not in vec.source
    xs = np.asarray([0.5, 0.0])
    assert np.allclose(vec(xs), [math.atanh(0.5), 0.0])


def test_protect_domains_atanh_dead_lanes() -> None:
    vec = array_vectorize.vectorize(
        support.make_fn("    return math.atanh(x) if -1 < x < 1 else 0.0"), protect_domains=True
    )
    with np.errstate(all="raise"):
        got = vec(np.asarray([2.0, 0.5, -2.0]))
    assert np.allclose(got, [0.0, math.atanh(0.5), 0.0])


# ------------------------------------------------------- live lanes stay exact


def test_protect_domains_never_changes_results() -> None:
    fn = support.make_fn("    if -1 <= x <= 1:\n        return math.asin(x)\n    return 0.0")
    vec = array_vectorize.vectorize(fn, protect_domains=True)
    xs = np.asarray([-2.0, -0.5, 0.0, 0.5, 2.0])
    expected = np.asarray([0.0, math.asin(-0.5), 0.0, math.asin(0.5), 0.0])
    assert np.allclose(vec(xs), expected)


def test_protect_domains_preserves_signed_zero() -> None:
    vec = array_vectorize.vectorize(
        support.make_fn("    return math.copysign(1.0, math.sqrt(x)) if x <= 0 else 5.0"),
        protect_domains=True,
    )
    assert vec(np.asarray([-0.0]))[0] == -1.0


def test_protect_domains_preserves_subnormal_live_lanes() -> None:
    vec = array_vectorize.vectorize(
        support.make_fn("    return math.log(x) if x > 0 else 0.0"), protect_domains=True
    )
    got = vec(np.asarray([1e-310]))
    assert got[0] == math.log(1e-310)


# --------------------------------------------------------- what is not clamped


def test_protect_domains_off_by_default() -> None:
    fn = support.make_fn("    if x > 0:\n        return math.sqrt(x)\n    return 0.0")
    vec = array_vectorize.vectorize(fn)
    assert "xp.maximum(x, 0.0)" not in vec.source


def test_protect_domains_top_level_calls_not_clamped() -> None:
    fn = support.make_fn("    return math.sqrt(abs(x))")
    vec = array_vectorize.vectorize(fn, protect_domains=True)
    assert "maximum" not in vec.source


def test_protect_domains_conditions_not_clamped() -> None:
    # a partial call inside a condition must not be clamped (results
    # never change); isnan(sqrt(x)) is a condition here
    fn = support.make_fn("    return 1.0 if math.isnan(math.sqrt(x)) else 0.0")
    vec = array_vectorize.vectorize(fn, protect_domains=True)
    with np.errstate(invalid="ignore"):
        got = vec(np.asarray([-1.0, 4.0]))
    assert np.array_equal(got, [1.0, 0.0])


# ----------------------------------------------------------- liveness analysis


def test_protect_domains_follows_branch_bindings() -> None:
    vec = array_vectorize.vectorize(
        support.make_fn(
            "    if x > 0:\n        y = math.sqrt(x)\n    else:\n        y = 0.0\n    return y"
        ),
        protect_domains=True,
    )
    with np.errstate(all="raise"):
        got = vec(np.asarray([-1.0, 1.0]))
    assert np.allclose(got, [0.0, 1.0])


def test_protect_domains_multiple_uses_or_liveness() -> None:
    # y_1 is used in two wheres with different (overlapping) conditions:
    # only lanes dead for BOTH uses are clampable; live lanes keep exact
    # values. Live region = (x > 0) or (x >= 0) = x >= 0; dead = x < 0.
    vec = array_vectorize.vectorize(
        support.make_fn(
            "    y = math.sqrt(x)\n"
            "    a = y if x > 0 else 0.0\n"
            "    b = y if x >= 0 else 1.0\n"
            "    return a + b"
        ),
        protect_domains=True,
    )
    with np.errstate(all="raise"):
        got = vec(np.asarray([-4.0, 4.0]))
    assert np.allclose(got, [1.0, 4.0])


def test_protect_domains_self_reference_guard() -> None:
    vec = array_vectorize.vectorize(
        support.make_fn(
            "    if x < 0:\n"
            "        return 0.0\n"
            "    y = math.sqrt(x)\n"
            "    return y if y > 1 else 0.0"
        ),
        protect_domains=True,
    )
    with np.errstate(all="raise"):
        got = vec(np.asarray([-1.0, 4.0]))
    assert np.allclose(got, [0.0, 2.0])


def test_protect_domains_forward_reference_guard() -> None:
    # the liveness condition references z, which is bound after the sqrt:
    # the clamp must be skipped (not crash), values stay exact
    vec = array_vectorize.vectorize(
        support.make_fn(
            "    y = math.sqrt(x)\n"
            "    if x > 0:\n"
            "        z = x\n"
            "    else:\n"
            "        z = 1.0\n"
            "    return y if z > 0 else 0.0"
        ),
        protect_domains=True,
    )
    with np.errstate(all="raise"):
        got = vec(np.asarray([0.0, 4.0]))
    assert np.allclose(got, [0.0, 2.0])


def test_protect_domains_loop_body_binding_guard() -> None:
    body = (
        "    y = math.sqrt(x)\n"
        "    for i in range(1):\n"
        "        c = x > 0\n"
        "    return y if c else 0.0"
    )
    plain = array_vectorize.vectorize(support.make_fn(body))(np.asarray([4.0]))
    protected = array_vectorize.vectorize(support.make_fn(body), protect_domains=True)(
        np.asarray([4.0])
    )
    assert plain[0] == 2.0
    assert protected[0] == 2.0


def test_protect_domains_propagates_to_helpers() -> None:
    # the helper's own internal where gets protected; note that a helper
    # called only inside a caller's where-branch cannot know its lanes are
    # dead (cross-function liveness is a documented limitation)
    fn = support.make_fn(
        "    return helper(x) * 2.0",
        extra="def helper(v):\n    return math.sqrt(v) if v > 0 else 0.0\n",
    )
    vec = array_vectorize.vectorize(fn, protect_domains=True)
    with np.errstate(all="raise"):
        got = vec(np.asarray([-1.0, 4.0]))
    assert np.allclose(got, [0.0, 4.0])
