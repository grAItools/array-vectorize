# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

import math
import warnings
import numpy as np
import array_api_strict as strict
from array_vectorize import vectorize

def nested(x):
    a=math.sqrt(x)
    s=0.0
    for i in range(1):
        for j in range(1):
            s=s+a
    return a if x>0 else s

with warnings.catch_warnings():
 warnings.simplefilter('ignore')
 for xp in [np,strict]:
  x=xp.asarray([-1.,0.,1.])
  for protect in [False,True]:
   try:
    v=vectorize(nested,protect_domains=protect)
    print(xp.__name__,protect,v(x))
    if protect: print(v.source)
   except Exception as e: print(type(e).__name__,str(e))
