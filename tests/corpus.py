"""Corpus of scalar functions for golden-source tests (plan T2).

Every function here must be defined at module level in this real file so
``inspect.getsource`` works. Golden outputs live in ``tests/golden/cases``.
"""

import math
from math import exp as fexp

import numpy as np

from vectorizer import vectorize

SCALE = 3.0
ARR = np.asarray([1.0, 2.0, 3.0])


def add(x, y):
    return x + y


def arith_ops(x, y):
    return (x + y - x * y) / (x * x + y * y)


def floor_mod_ops(x):
    return (x // 2) + (x % 3)


def pow_op(x):
    return x**2 + 2.0**x


def bitwise_ops(x, y):
    return (x & y) | (x ^ 3) << 1 >> 2


def unary_ops(x):
    return -x + +x - ~x


def math_calls(x):
    return math.sqrt(x) + math.exp(x) + math.log1p(x) + math.sin(x)


def math_two_arg(x, y):
    return math.atan2(x, y) + math.hypot(x, y) + math.copysign(x, y)


def math_floor_ceil(x):
    return math.floor(x) + math.ceil(x)


def from_import(x):
    return fexp(x)


def minmax(x, y):
    return min(x, y) + max(x, y, 3.0)


def casts(x):
    return int(x) + float(x) + bool(x) + math.trunc(x)


def abs_round(x):
    return abs(x) + round(x)


def constants(x):
    return x * math.pi + math.e + math.inf - math.nan


def compare_chain(x):
    return 0 < x < 1


def compare_ops(x, y):
    return (x == y) + (x != y) + (x < y) + (x <= y) + (x > y) + (x >= y)


def boolop_logical(x, y):
    return ((x > 0) and (y > 0)) or not (x < 0)


def boolop_numeric(x, y):
    return x and y


def ternary(x):
    return x * 2 if x > 0 else x / 2


def ssa_rebind(x):
    y = x + 1
    y = y * 2
    y += 3
    return y


def closure_scalar(x):
    return x * SCALE


def closure_array(x):
    return x + ARR


def relu(x):
    if x > 0:
        return x
    return 0.0


def relu_with_else(x):
    if x > 0:  # noqa: SIM108
        r = x
    else:
        r = 0.0
    return r


def psi(x):
    if x < 0:
        return 0.0
    return x * math.exp(-x)


def clamp(x, lo=0.0, hi=1.0):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def piecewise(x):
    if x < -1:
        return -1.0
    elif x > 1:
        return 1.0
    return x


def merge_with_prior(x):
    r = 0.0
    if x > 0:
        r = x
    return r


def nested_early_returns(x):
    if x > 0:
        if x > 10:
            return 1.0
        return 2.0
    return 3.0


def both_branches_assign(x):
    if x > 0:
        y = x
        z = 1.0
    else:
        y = -x
        z = 2.0
    return y + z


def kwonly_defaults(x, *, scale=2.0, bias=0.5):
    return x * scale + bias


def reserved_param(xp):
    return xp + 1


def cse_opportunity(x):
    return math.sqrt(x) * math.sqrt(x) + math.sqrt(x)


def complex_expr(x, y):
    t = math.exp(-(x * x + y * y))
    return t / (1.0 + t)


def loop_accumulate(x):
    s = 0.0
    for i in range(4):  # noqa: B007
        s = s + x
    return s


def loop_start_stop_step(x):
    s = 1.0
    for i in range(2, 8, 3):
        s = s * i
    return s + x


def loop_two_carried(x):
    a = 0.0
    b = 1.0
    for i in range(3):  # noqa: B007
        a = a + x
        b = b * 2.0
    return a + b


def loop_with_branch(x):
    s = 0.0
    for i in range(4):  # noqa: B007
        if x > 0:  # noqa: SIM108
            s = s + x
        else:
            s = s - 1.0
    return s


def nested_loops(x):
    s = 0.0
    for i in range(3):  # noqa: B007
        for j in range(2):  # noqa: B007  # noqa: B007
            s = s + x
    return s


def helper_inner(y):
    return y * y + 1.0


def helper_outer(x):
    return helper_inner(x) + helper_inner(x * 2.0)


def loop_helper_caller(x):
    s = 0.0
    for i in range(3):  # noqa: B007
        s = s + helper_inner(x)
    return s


GOLDEN_NAMES = [
    "loop_accumulate",
    "loop_start_stop_step",
    "loop_two_carried",
    "loop_with_branch",
    "nested_loops",
    "helper_outer",
    "loop_helper_caller",
    "add",
    "arith_ops",
    "floor_mod_ops",
    "pow_op",
    "bitwise_ops",
    "unary_ops",
    "math_calls",
    "math_two_arg",
    "math_floor_ceil",
    "from_import",
    "minmax",
    "casts",
    "abs_round",
    "constants",
    "compare_chain",
    "compare_ops",
    "boolop_logical",
    "boolop_numeric",
    "ternary",
    "ssa_rebind",
    "closure_scalar",
    "closure_array",
    "relu",
    "relu_with_else",
    "psi",
    "clamp",
    "piecewise",
    "merge_with_prior",
    "nested_early_returns",
    "both_branches_assign",
    "kwonly_defaults",
    "reserved_param",
    "cse_opportunity",
    "complex_expr",
]

__all__ = ["GOLDEN_NAMES", "vectorize"]
