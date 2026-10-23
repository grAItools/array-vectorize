# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

import concurrent.futures
import importlib.util
import itertools
import math
import pathlib
import tempfile
import threading
import warnings

import numpy as np
import array_api_strict as strict
from array_vectorize import vectorize, VectorizationError
from array_vectorize import api

tmp = tempfile.TemporaryDirectory(prefix='reconcile_subjects_')
seq = itertools.count()

def module(source):
    path = pathlib.Path(tmp.name) / f'subject_{next(seq)}.py'
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def observe(label, fn):
    try:
        print(label, '=>', repr(fn()))
    except Exception as exc:
        print(label, '=>', type(exc).__name__, str(exc))

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for xp in (np, strict):
        print('\nBACKEND', xp.__name__)
        for body in ['a + -1', 'a * -1']:
            f = module(f'def f(x):\n    a = min(x, True)\n    return {body}\n').f
            observe('runtime negative ' + body, lambda f=f, body=body: vectorize(f)(xp.asarray([0 if '+' in body else 1], dtype=xp.uint64)))
        for name, src, args in [
            ('not numeric', 'def f(x):\n    return not x\n', (xp.asarray([0.,2.]),)),
            ('loop raw scalar', 'def f(x):\n    s=0.0\n    for i in range(3):\n        s=s+(i//2)+(i%2)+(i**2)\n    return s+x\n', (xp.asarray([0.]),)),
            ('scalar flag', 'def f(x,flag):\n    if flag>0:\n        return x\n    return -x\n', (xp.asarray([1.,2.]), 1.)),
            ('floor large integer', 'import math\ndef f(x):\n    return math.floor(x)\n', (xp.asarray([2**53+1], dtype=xp.int64),)),
            ('abs int64 minimum', 'def f(x):\n    return abs(x)\n', (xp.asarray([-2**63], dtype=xp.int64),)),
            ('integer remainder zero', 'def f(x):\n    return x%0\n', (xp.asarray([4], dtype=xp.int64),)),
            ('integer floordiv zero', 'def f(x):\n    return x//0\n', (xp.asarray([4], dtype=xp.int64),)),
        ]:
            f = module(src).f
            observe(name, lambda f=f, args=args: vectorize(f)(*args))
        decorated = module('import functools\ndef deco(fn):\n    @functools.wraps(fn)\n    def wrapper(x):\n        return fn(x)+100.0\n    return wrapper\n@deco\ndef f(x):\n    return x*2.0\n')
        observe('decorated scalar expected106', lambda: decorated.f(3.))
        observe('decorated actual', lambda: vectorize(decorated.f)(xp.asarray([3.])))
        complex_pow = module('def f(x):\n    return x**0.5\n').f
        observe('verify complex-producing pow', lambda: vectorize(complex_pow, verify=(xp.asarray([-8.]),)))
        arithmetic = module('import math\ndef f(x):\n    return 1/x\n').f
        observe('verify zero division documented nan expectation', lambda: vectorize(arithmetic, verify=(xp.asarray([0.]),)))
    huge = module('def f(x):\n    return x+int(1e30)\n')
    observe('correct huge FLOAT oracle', lambda: huge.f(1.0))
    observe('correct huge FLOAT vector', lambda: vectorize(huge.f)(np.asarray([1.0])))
    for value in ['-9223372036854775808', '9223372036854775808']:
        f = module(f'def f(x):\n    return {value}\n').f
        observe('literal '+value, lambda f=f: vectorize(f))
    badcast = module('def f(x):\n    return int(x)\n').f
    observe('cast float result fits uint64', lambda: vectorize(badcast)(np.asarray([float(2**63)])))

class ScalarCallable:
    def __call__(self, x):
        return x + 1

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    observe('fallback callable object', lambda: vectorize(ScalarCallable(), fallback=True)(np.asarray([1])))

mod = module('def helper(y):\n    return y+1\ndef first(x):\n    return helper(x)\ndef second(x):\n    return helper(x)\n')
original = api._vectorize_strict
entered = threading.Event()
release = threading.Event()

def pause_helper(func, **kwargs):
    if func is mod.helper:
        entered.set()
        if not release.wait(10):
            raise RuntimeError('concurrency probe timed out')
    return original(func, **kwargs)

api._vectorize_strict = pause_helper
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(vectorize, mod.first)
        if not entered.wait(10):
            raise RuntimeError('helper did not reach pause')
        try:
            observe('concurrent independent compile', lambda: vectorize(mod.second))
        finally:
            release.set()
        observe('first compile completes', lambda: future.result()(np.asarray([1])))
finally:
    api._vectorize_strict = original
