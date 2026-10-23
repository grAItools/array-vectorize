# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

import math
import warnings
import numpy as np
import array_api_strict as strict
from array_vectorize import vectorize


def plus(x):
    return +x

def invert(x):
    return ~x

def neg(x):
    return -x

def abst(x):
    return abs(x)

def rnd(x):
    return round(x)

def floor(x):
    return math.floor(x)

def ceil(x):
    return math.ceil(x)

def pown(x):
    return x ** -1

def powv(x,y):
    return x ** y

def eq(x,y):
    return x == y

def lt(x,y):
    return x < y

def add(x,y):
    return x + y

def sub(x,y):
    return x - y

def mod(x,y):
    return x % y

def div(x,y):
    return x / y

def mixed(x):
    return x == 9007199254740993

with warnings.catch_warnings():
 warnings.simplefilter('ignore')
 for xp in (np,strict):
  print('\nBACKEND',xp.__name__)
  for f in [plus,invert,neg,abst,rnd,floor,ceil,pown]:
   try:
    v=vectorize(f)
    a=xp.asarray([False,True],dtype=xp.bool)
    print(f.__name__, 'bool',v(a))
   except Exception as e: print(f.__name__,type(e).__name__,str(e))
  for f in [eq,lt,add,sub,mod,div]:
   for dt1,vals1,dt2,vals2 in [('int64',[2**53+1,2**63-1],'float64',[float(2**53),float(2**63)]),('uint64',[2**63+1,2**64-1],'int64',[2**63-1,-1])]:
    try:
     a=xp.asarray(vals1,dtype=getattr(xp,dt1)); b=xp.asarray(vals2,dtype=getattr(xp,dt2))
     exp=[f(x,y) for x,y in zip(vals1,vals2)]
     print(f.__name__,dt1,dt2,'expected',exp,'actual',vectorize(f)(a,b))
    except Exception as e: print(f.__name__,dt1,dt2,type(e).__name__,str(e))
  try: print('mixed float literal expected',[mixed(float(2**53))], 'actual',vectorize(mixed)(xp.asarray([float(2**53)],dtype=xp.float64)))
  except Exception as e: print(type(e).__name__,str(e))
