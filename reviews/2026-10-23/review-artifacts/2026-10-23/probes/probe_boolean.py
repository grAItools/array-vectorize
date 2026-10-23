# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

import numpy as np
import array_api_strict as strict
from array_vectorize import vectorize

def add(x,y):
    return x+y

def sub(x,y):
    return x-y

def mul(x,y):
    return x*y

def inv_cmp(x):
    return ~(x>0)

def plus_cmp(x):
    return +(x>0)

for xp in [np,strict]:
 print(xp.__name__)
 for f in [add,sub,mul]:
  try: print(f.__name__,vectorize(f)(xp.asarray([True,False]),xp.asarray([True,True])))
  except Exception as e: print(f.__name__,type(e).__name__,str(e))
 for f in [inv_cmp,plus_cmp]:
  try: print(f.__name__,[f(x) for x in [-1,1]],vectorize(f)(xp.asarray([-1,1])))
  except Exception as e: print(f.__name__,type(e).__name__,str(e))
