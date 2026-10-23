# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

import math
def f0(x):
    return [x]
def f1(x):
    return x[0]
def f2(x):
    return x.real
def f3(x):
    return None
def f4(x):
    return (x,x)
def f5(x):
    while x>0:
        x-=1
    return x
def f6(x):
    assert x>0
    return x
def f7(x):
    try:
        return x
    except Exception:
        return 0
def f8(x):
    return math.sqrt(x,x)
def f9(x):
    for i in range(1.5):
        x+=i
    return x
def f10(x):
    if x:
        return 1
    return 0
def f11(x):
    return missing(x)
