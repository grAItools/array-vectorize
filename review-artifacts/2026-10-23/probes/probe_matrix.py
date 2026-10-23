# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

import ast
import collections
import importlib.util
import math
from pathlib import Path
import warnings
import numpy as np
import array_api_strict as strict
from array_vectorize import vectorize

unary={'neg':'-x','pos':'+x','inv':'~x','abs':'abs(x)','round':'round(x)','floor':'math.floor(x)','ceil':'math.ceil(x)','trunc':'math.trunc(x)','int':'int(x)','float':'float(x)','bool':'bool(x)','sqrt':'math.sqrt(x)','not':'not x','negpow':'x ** -1','localnegpow':'int(x) ** -1','minzero':'min(x,0)','maxzero':'max(x,0)'}
binary={'add':'x+y','sub':'x-y','mul':'x*y','div':'x/y','floor_div':'x//y','mod':'x%y','eq':'x==y','lt':'x<y','gt':'x>y','and':'x and y','or':'x or y','min':'min(x,y)','max':'max(x,y)'}
functions_path=Path('/tmp/probe_matrix_functions.py')
functions_path.write_text('import math\n'+''.join(f'def f_{name}(x):\n    return {expr}\n' for name,expr in unary.items())+''.join(f'def b_{name}(x,y):\n    return {expr}\n' for name,expr in binary.items()))
spec=importlib.util.spec_from_file_location('matrix_functions',functions_path)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
values={'bool':[False,True],'int64':[-2**63,-2**53-1,-1,0,1,2**53+1,2**63-1],'uint64':[0,1,2**53+1,2**63,2**63+1,2**64-1],'float64':[0.,-0.,1.,-1.,5e-324,-5e-324,float('inf'),-float('inf'),float('nan'),float(2**53),float(2**63)]}
count=0; skip=0; mismatches=collections.Counter(); errors=collections.Counter(); examples={}

def same(expected,actual):
    if isinstance(expected,float) and math.isnan(expected): return isinstance(actual,float) and math.isnan(actual)
    if expected!=actual: return False
    if isinstance(expected,float) and expected==0: return math.copysign(1,expected)==math.copysign(1,actual)
    return True

with warnings.catch_warnings():
 warnings.simplefilter('ignore')
 for xp in [np,strict]:
  for arity,fs in [(1,unary),(2,binary)]:
   for name in fs:
    f=getattr(m,('f_' if arity==1 else 'b_')+name)
    try: v=vectorize(f)
    except Exception as exc:
     print('DECORATION ERROR',name,type(exc).__name__,str(exc)); continue
    for dt,xs in values.items():
     for x in xs:
      ygroups=values.items() if arity==2 else [('none',[None])]
      for dt2,ys in ygroups:
       for y in ys:
        args=[x]+([y] if arity==2 else [])
        try:
         expected=f(*args)
         if isinstance(expected,int) and not -2**63<=expected<=2**64-1: skip+=1; continue
        except (ArithmeticError,ValueError,TypeError): skip+=1; continue
        count+=1
        arr=[xp.asarray([x],dtype=getattr(xp,dt))]+([xp.asarray([y],dtype=getattr(xp,dt2))] if arity==2 else [])
        key=(xp.__name__,name,dt,dt2)
        try:
         result=np.asarray(v(*arr)).item()
         if not same(expected,result):
          mismatches[key]+=1; examples.setdefault(key,(args,expected,result))
        except Exception as e:
         errors[key]+=1; examples.setdefault(key,(args,expected,type(e).__name__+': '+str(e)))
print('checked',count,'scalar-exception/out-of-range skips',skip,'mismatch lanes',sum(mismatches.values()),'runtime errors',sum(errors.values()))
print('mismatch groups',len(mismatches),'error groups',len(errors))
for bucket,counter in [('MISMATCH',mismatches),('ERROR',errors)]:
 for key,n in counter.items():
  if key[1] in ['min','max','mod','floor_div','neg','negpow','localnegpow','sqrt','round','int','trunc']:
   print(bucket,key,n,examples[key])
Path('/tmp/probe_matrix_results.txt').write_text('\n'.join(f'{bucket} {key} {n} {examples[key]}' for bucket,counter in [('MISMATCH',mismatches),('ERROR',errors)] for key,n in counter.items()))
