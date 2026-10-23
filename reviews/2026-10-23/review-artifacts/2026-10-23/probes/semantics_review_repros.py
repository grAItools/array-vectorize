# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

"""Run: UV_CACHE_DIR=/tmp/array-vectorize-review-uv uv run --no-sync python /tmp/semantics_review_repros.py."""
import math
import warnings
import numpy as np
import array_api_strict as strict
from array_vectorize import vectorize

def boolean_add(x, y):
    return x + y

def boolean_invert(x):
    return ~(x > 0)

def integer_sub(x, y):
    return x - y

def compare_large_literal(x):
    return x == 9007199254740993

def rounding_then_bitwise(x):
    return math.floor(x) & 1

def negative_power(x):
    return x ** -1

def unsigned_int_cast(x):
    return int(x)

def minimum_zero(x, y):
    return math.copysign(1.0, min(x, y))

def nested_protection(x):
    a = math.sqrt(x)
    s = 0.0
    for i in range(1):
        for j in range(1):
            s = s + a
    return a if x > 0 else s

def range(n):
    return (0, 10)

def shadowed_range(x):
    for i in range(2):
        x += i
    return x

def invalid_constant_domain(x):
    return math.sqrt(-1.0) + x

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for xp in (np, strict):
        print('\nBACKEND:', xp.__name__)
        cases = [
            (boolean_add, (([True], 'bool'), ([True], 'bool')), [2], False),
            (boolean_invert, (([-1, 1], 'int64'),), [-1, -2], False),
            (integer_sub, (([2**63 + 1], 'uint64'), ([2**63 - 1], 'int64')), [2], False),
            (compare_large_literal, (([float(2**53)], 'float64'),), [False], False),
            (rounding_then_bitwise, (([1.5], 'float64'),), [1], False),
            (negative_power, (([2], 'int64'),), [0.5], False),
            (unsigned_int_cast, (([2**63 + 1], 'uint64'),), [2**63 + 1], False),
            (minimum_zero, (([0.0, -0.0], 'float64'), ([-0.0, 0.0], 'float64')), [1., -1.], False),
            (nested_protection, (([-1., 1.], 'float64'),), [float('nan'), 1.], True),
            (shadowed_range, (([1, 2], 'int64'),), [11, 12], False),
            (invalid_constant_domain, (([1.], 'float64'),), [float('nan')], False),
        ]
        for f, specs, expected, protect in cases:
            try:
                args = [xp.asarray(values, dtype=getattr(xp, dt)) for values, dt in specs]
                actual = vectorize(f, protect_domains=protect)(*args)
                print(f.__name__, 'expected:', expected, 'actual:', actual)
            except Exception as error:
                print(f.__name__, 'expected:', expected, 'actual:', type(error).__name__, str(error))
