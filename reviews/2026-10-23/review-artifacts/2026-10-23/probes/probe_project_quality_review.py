# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

import random
import numpy as np
from array_vectorize import fuzz
from array_vectorize import vectorize
from array_vectorize.verify import verify_match
import array_api_strict as xps

def constant(x):
    return 2.0

old = fuzz.vectorize
try:
    fuzz.vectorize = lambda fn: lambda x: np.ones_like(x, dtype=bool)
    print('fuzzer wrong numeric-to-bool', fuzz._check_case(random.Random(0), 'def constant(x): return 2.0', constant))
    def crash(x):
        raise TypeError('genuine runtime failure')
    fuzz.vectorize = lambda fn: crash
    print('fuzzer unconditional TypeError', fuzz._check_case(random.Random(0), 'def constant(x): return 2.0', constant))
finally:
    fuzz.vectorize = old

fuzz.main(['--seed', '0', '--cases', '1', '--seconds', '0'])

def compare_large(x):
    return x == 9007199254740993

for xp in (np, xps):
    xs = xp.asarray([float(2**53)], dtype=xp.float64)
    try:
        v = vectorize(compare_large, verify=(xs,))
        print(xp.__name__, 'public verify missed comparison', v(xs), 'expected', compare_large(float(2**53)))
    except Exception as exc:
        print(xp.__name__, 'public verify caught comparison', type(exc).__name__)

def mixed(x):
    return 1.5 if x > 0 else 2

for xp in (np, xps):
    try:
        v = vectorize(mixed, verify=(xp.asarray([1.0, -1.0], dtype=xp.float64),))
        print(xp.__name__, 'mixed result verify okay')
    except Exception as exc:
        print(xp.__name__, 'mixed result verify', type(exc).__name__, str(exc))
