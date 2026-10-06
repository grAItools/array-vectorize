# Quickstart

## Your first vectorized function

```python
import math
import numpy as np
from array_vectorize import vectorize

@vectorize
def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)

x = np.asarray([-3.0, -1.0, 0.0, 1.0, 3.0])
sigmoid(x)
```

Branches merge into `xp.where` selects, so both sides must be defined for
every lane (see [semantics](semantics.md#eager-branches-dead-lanes)).

## Inspecting the output

```python
print(sigmoid.source)         # the generated Python source
inspect.getsource(sigmoid)    # also works
help(sigmoid)                 # original docstring (prefixed) + scalar source in Notes
```

The generated function is plain Array API code — you can paste it into
your own codebase, profile it, or edit it.

## Running on other backends

The generated function takes its namespace from its arguments:

```python
import array_api_strict as xps
sx = xps.asarray([-3.0, -1.0, 0.0, 1.0, 3.0])
sigmoid(sx)                   # same values, strict backend
```

Torch, JAX, and CuPy work the same way (their arrays are
standard-compliant via `array_api_compat`).

### Pinning the namespace

Instead of detecting the backend from the arguments, you can pin it up
front — the generated code binds `xp` directly to that namespace and never
inspects its arguments (so all-scalar calls work too). Passing compatible
arguments is then your responsibility:

```python
sigmoid_np = vectorize(sigmoid, namespace=np)   # pinned at decoration time
sigmoid_np(x)                                   # runs on NumPy

sigmoid_xps = sigmoid.with_namespace(xps)       # a NEW callable, pinned to strict
sigmoid_xps(sx)
```

`with_namespace` returns a new callable (the original is untouched),
keeps `.source` and the signature, and chains. Note that pinned generated
source has no `array_namespace` import — if you paste it elsewhere, pass
the hidden `_namespace=` keyword or rebind `xp`.

## Options

```python
vectorize(f)                        # strict vectorization (default)
vectorize(f, strict=False)          # fall back to an element loop on failure
vectorize(f, fallback=True)         # same, with an explicit UserWarning
vectorize(f, protect_domains=True)  # clamp partial functions in dead lanes
vectorize(f, verify=example_args)   # differential check at generation time
```

- Vectorization happens eagerly at decoration time (fail fast).
- Signature, defaults, and kwarg names are preserved.
- `vectorize(vectorize(f))` is idempotent.

## When a function is rejected

```python
@vectorize
def bad(x):
    while x > 0:          # while loops are not supported
        x = x - 1
    return x
# VectorizationError: <file>:3:5: 'while' loops are not supported
#                    (supported: 'for i in range(N)' with constant N)
```

Every rejection carries line/column information and all violations are
collected at once. See the [supported subset](supported-subset.md) for
the full list.
