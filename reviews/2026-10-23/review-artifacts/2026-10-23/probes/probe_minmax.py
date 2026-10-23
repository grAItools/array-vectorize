# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

import math
import numpy as np
import array_api_strict as strict
from array_vectorize import vectorize

def mn(x,y):
    return min(x,y)

def mx(x,y):
    return max(x,y)

def floormod(x):
    return math.floor(x) % 2

for xp in [np,strict]:
 print(xp.__name__)
 for f in [mn,mx]:
  for a,b in [([0.,-0.],[-0.,0.]),([2**53+1],[float(2**53)]),([2**63+1],[2**63-1])]:
   try:
    actual=vectorize(f)(xp.asarray(a),xp.asarray(b))
    print(f.__name__,'expected',[(f(x,y),math.copysign(1,f(x,y))) for x,y in zip(a,b)],'actual',actual,'signbit',xp.signbit(actual))
   except Exception as e: print(type(e).__name__,str(e))
