# array-vectorize: compile scalar Python functions into exact Array API functions.
# Copyright (c) 2026 grAItools
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE for the full license text.

import importlib.util
from pathlib import Path
from array_vectorize import vectorize, VectorizationError
bodies=['return [x]','return x[0]','return x.real','return None','return (x,x)','while x>0:\n    x-=1\nreturn x','assert x>0\nreturn x','try:\n    return x\nexcept Exception:\n    return 0','return math.sqrt(x,x)','for i in range(1.5):\n    x+=i\nreturn x','if x:\n    return 1\nreturn 0','return missing(x)']
text='import math\n'+''.join(f'def f{i}(x):\n'+''.join('    '+s+'\n' for s in body.splitlines()) for i,body in enumerate(bodies))
p=Path('/tmp/rejection_functions.py');p.write_text(text)
spec=importlib.util.spec_from_file_location('rejections',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for i in range(len(bodies)):
 try: vectorize(getattr(m,f'f{i}')); print(i,'ACCEPTED')
 except VectorizationError as error: print(i,'VectorizationError',bool(error.diagnostics),str(error).splitlines()[0])
 except Exception as error: print(i,type(error).__name__,str(error))
