# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

import numpy as np
import array_api_strict as strict
from array_vectorize import vectorize

def h(y):
    return y*y

def f(x):
    return h(2)+x

for xp in [np,strict]:
    try: print(xp.__name__, vectorize(f)(xp.asarray([1,2])))
    except Exception as e: print(xp.__name__,type(e).__name__,str(e))
