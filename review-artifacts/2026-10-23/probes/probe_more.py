# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

import math
import warnings
import numpy as np
import array_api_strict as strict
from array_vectorize import vectorize

def zeros(x):
    return math.atan2(x, 0.0)-math.atan2(x,-0.0)

def floorbit(x):
    return math.floor(x) & 1

def roundbit(x):
    return round(x) & 1

def ceilbit(x):
    return math.ceil(x) & 1

def pown(x):
    return x**-1

def pownlocal(x):
    n=-1
    return int(x)**n

def booland(x,y):
    return x and y

def boolor(x,y):
    return x or y

def boolnot(x):
    return not x

with warnings.catch_warnings():
 warnings.simplefilter('ignore')
 for xp in [np,strict]:
  print(xp.__name__)
  for f,vals,dt in [(zeros,[0.,-0.,1.,-1.],xp.float64),(floorbit,[1.5,-1.5],xp.float64),(roundbit,[1.5,-1.5],xp.float64),(ceilbit,[1.5,-1.5],xp.float64),(pown,[1,2,-2],xp.int64),(pownlocal,[1,2,-2],xp.int64),(boolnot,[True,False],xp.bool)]:
   try:
    v=vectorize(f)
    print(f.__name__,'expected',[f(x) for x in vals],'actual',v(xp.asarray(vals,dtype=dt)))
    if f is zeros: print(v.source)
   except Exception as e: print(f.__name__,type(e).__name__,str(e))
  for f in [booland,boolor]:
   try: print(f.__name__,vectorize(f)(xp.asarray([True,False]),xp.asarray([False,True])))
   except Exception as e: print(f.__name__,type(e).__name__,str(e))
