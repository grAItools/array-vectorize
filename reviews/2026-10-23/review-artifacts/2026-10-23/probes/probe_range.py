# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

import numpy as np
import array_api_strict as strict
from array_vectorize import vectorize

def range(n):
    return (0,10)

def f(x):
    for i in range(2):
        x+=i
    return x

for xp in [np,strict]:
    print(xp.__name__,[f(v) for v in [1,2]],vectorize(f)(xp.asarray([1,2])))
