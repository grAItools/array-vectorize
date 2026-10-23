# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

import math
def f_neg(x):
    return -x
def f_pos(x):
    return +x
def f_inv(x):
    return ~x
def f_abs(x):
    return abs(x)
def f_round(x):
    return round(x)
def f_floor(x):
    return math.floor(x)
def f_ceil(x):
    return math.ceil(x)
def f_trunc(x):
    return math.trunc(x)
def f_int(x):
    return int(x)
def f_float(x):
    return float(x)
def f_bool(x):
    return bool(x)
def f_sqrt(x):
    return math.sqrt(x)
def f_not(x):
    return not x
def f_negpow(x):
    return x ** -1
def f_localnegpow(x):
    return int(x) ** -1
def f_minzero(x):
    return min(x,0)
def f_maxzero(x):
    return max(x,0)
def b_add(x,y):
    return x+y
def b_sub(x,y):
    return x-y
def b_mul(x,y):
    return x*y
def b_div(x,y):
    return x/y
def b_floor_div(x,y):
    return x//y
def b_mod(x,y):
    return x%y
def b_eq(x,y):
    return x==y
def b_lt(x,y):
    return x<y
def b_gt(x,y):
    return x>y
def b_and(x,y):
    return x and y
def b_or(x,y):
    return x or y
def b_min(x,y):
    return min(x,y)
def b_max(x,y):
    return max(x,y)
