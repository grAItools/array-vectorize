# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

import inspect
import weakref
import gc
import warnings
import numpy as np
import array_api_strict as xps
from array_vectorize import vectorize
from array_vectorize.verify import verify_match

def attempt(label, action):
    try:
        print(label, action())
    except Exception as exc:
        print(label, type(exc).__name__, str(exc))

def make_a():
    def same_name(x):
        return x + 1
    return same_name

def make_b():
    def same_name(x):
        return x * 9
    return same_name

a = vectorize(make_a())
before = inspect.getsource(a)
b = vectorize(make_b())
print('linecache source changed', before != inspect.getsource(a))
print('linecache a now shows b', inspect.getsource(a) == inspect.getsource(b))
print('linecache a source attribute still original', 'x + 1' in a.source)

def reserved(array_namespace):
    return array_namespace + 1

v = vectorize(reserved)
print('reserved advertised signature', inspect.signature(v))
attempt('reserved keyword', lambda: v(array_namespace=np.array([2])))
print('reserved positional', v(np.array([2])))

SCALE = 2
def scaled(x):
    return x * SCALE

v = vectorize(scaled)
SCALE = 3
print('frozen original', v(np.array([1])))
print('new namespace pin', v.with_namespace(np)(np.array([1])))

def defaults(x, y=2):
    return x + y

defaults.__defaults__ = (7,)
v = vectorize(defaults)
print('defaults expected', defaults(1), 'got', v(np.array([1])), 'signature', inspect.signature(v))

def identity(x):
    return x

for xp in (np, xps):
    print('backend', xp.__name__)
    attempt('verify integer lie', lambda: verify_match(lambda x: x + 1, identity, (xp.asarray([2**53], dtype=xp.int64),)))
    attempt('verify zero sign lie', lambda: verify_match(lambda x: -x, identity, (xp.asarray([0.0], dtype=xp.float64),)))

def unsupported(x):
    while x < 0:
        x += 1
    return x

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    v = vectorize(unsupported, namespace=np, fallback=True)
attempt('pinned fallback scalar', lambda: v(-1))

def local_func():
    def ephemeral(x):
        return x + 1
    return ephemeral

f = local_func()
ref = weakref.ref(f)
v = vectorize(f)
del f, v
gc.collect()
print('cache retains discarded original', ref() is not None)
