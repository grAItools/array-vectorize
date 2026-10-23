# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

import math
import warnings
import numpy as np
import array_api_strict as strict
from array_vectorize import vectorize

def sqrt_const(x):
    return math.sqrt(-1.)+x

def pow_const(x):
    return math.pow(-1.,0.5)+x

def const_cond(x):
    return math.sqrt(x) if True else 0.

def loop_cond(x):
    y=x
    for i in range(1):
        y=math.sqrt(x) if i==0 else 0.
    return y

with warnings.catch_warnings():
 warnings.simplefilter('ignore')
 for xp in [np,strict]:
  for f in [sqrt_const,pow_const,const_cond,loop_cond]:
   for protect in [False,True]:
    try: print(xp.__name__,f.__name__,protect,vectorize(f,protect_domains=protect)(xp.asarray([1.,2.])))
    except Exception as e: print(xp.__name__,f.__name__,protect,type(e).__name__,str(e))
