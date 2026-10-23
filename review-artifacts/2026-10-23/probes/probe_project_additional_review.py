# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

import inspect
import warnings
import numpy as np
import torch
import array_api_compat.torch as txp
from array_vectorize import vectorize
from array_vectorize import VectorizationError

def fallback_identity(x):
    while x < 0:
        x += 1
    return x

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    v = vectorize(fallback_identity, fallback=True, namespace=txp)
x = torch.tensor([1.0000000001], dtype=torch.float64)
got = v(x)
print('fallback float64 expected', x.item(), 'got', got.item(), 'dtype', got.dtype)

def ratio(x, y):
    return x / y

v = vectorize(ratio)
ns = {}
exec(v.source, ns)
try:
    ns['ratio_vec'](np.asarray([1]), np.asarray([2]))
except Exception as exc:
    print('pasted source', type(exc).__name__, str(exc))

def literal_default_negative(x, y=-1):
    return x + y

try:
    vectorize(literal_default_negative)
except VectorizationError as exc:
    print('negative literal default', str(exc), 'diagnostics', exc.diagnostics)

def rejected(x):
    while x > 0:
        x -= 1
    return x

try:
    vectorize(rejected)
except VectorizationError as exc:
    print('diagnostic actual location', inspect.getsourcelines(rejected)[1])
    print(str(exc))
