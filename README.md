# array-vectorize

`vectorize(f)` compiles an inspectable **scalar** Python function into a
**semantically equivalent function over arrays** that uses only the
[Python Array API standard](https://data-apis.org/array-api/latest/) — one
generated function runs on NumPy, PyTorch, JAX, CuPy, `array-api-strict`,
and any other standard-compliant backend. Unlike `numpy.vectorize` (a
Python-level element loop), the result is genuine vectorized code, and the
generated source is readable and inspectable.

```python
import math
import numpy as np
from array_vectorize import vectorize

@vectorize
def psi(x):
    if x < 0:
        return 0.0
    return x * math.exp(-x)

x = np.asarray([-2.0, -0.5, 0.0, 1.0, 3.0])
psi(x)              # array([0.        , 0.        , 0.        , 0.36787944, 0.14936121])

print(psi.source)
# from array_api_compat import array_namespace
#
# def psi_vec(x):
#     """
#     (array-vectorized) no docstring on the scalar original.
#
#     Notes:
#         Vectorized by array-vectorize from this scalar original::
#
#             @vectorize
#             def psi(x):
#                 if x < 0:
#                     return 0.0
#                 return x * math.exp(-x)
#     """
#     xp = array_namespace(x)
#     return xp.where(x < 0, 0.0, x * xp.exp(xp.astype(xp.asarray(-x), xp.float64)))
```

The same `psi` runs unchanged on torch tensors, JAX arrays, CuPy arrays, or
`array_api_strict` arrays.

## Installation

```bash
pip install "array-vectorize @ git+https://github.com/grAItools/array-vectorize.git@v0.1.0"
```

Requires Python >= 3.12. The only runtime dependency is `array-api-compat`.
Releases are tagged `vX.Y.Z` (see the
[changelog](https://github.com/grAItools/array-vectorize/blob/main/CHANGELOG.md));
the latest state of `main` installs without the `@tag`. The package is not
on PyPI yet; once published, plain `pip install array-vectorize` will work.
For development from a checkout, see
[Development](https://grAItools.github.io/array-vectorize/development/)
(`make install`, then `make check`).

## What it does (and refuses to do)

- **Reject over miscompile** — anything outside the supported subset raises
  `VectorizationError` with a precise diagnostic (line/column + caret),
  never silent wrong results.
- **Exact semantics** — `and`/`or`/`not`, `min`/`max`, boolean arithmetic,
  and mixed uint64/int64 arithmetic match scalar Python wherever a single
  dtype can hold the result; the few inherent differences (IEEE values
  instead of `ZeroDivisionError`, eagerly evaluated branches) are
  documented.
- **Inspectable output** — `vec.source` and `inspect.getsource(vec)` show
  the generated function; tracebacks point at its real lines; the docstring
  keeps the original's documentation (summary prefixed
  `(array-vectorized)`) with the scalar source in a `Notes:` section, and
  the scalar original grows an `__array_vectorized__` back-reference.
- **Options** — `strict=False`/`fallback=True` (element-loop fallback),
  `protect_domains=True`, `verify=example_args`, `namespace=xp` (pinned
  backend).

## Documentation

The full documentation is at <https://grAItools.github.io/array-vectorize/>
(sources in [`docs/`](docs/); interactive marimo notebooks in
[`examples/notebooks/`](examples/notebooks/)):

- [Quickstart](https://grAItools.github.io/array-vectorize/quickstart/)
- [Supported subset](https://grAItools.github.io/array-vectorize/supported-subset/)
  — every accepted construct and its Array API lowering
- [Semantics & divergences](https://grAItools.github.io/array-vectorize/semantics/)
- [Public API](https://grAItools.github.io/array-vectorize/api/)
- [Architecture](https://grAItools.github.io/array-vectorize/architecture/)
- [Development](https://grAItools.github.io/array-vectorize/development/)
  — `make install`, then `make check`
- [Changelog](https://github.com/grAItools/array-vectorize/blob/main/CHANGELOG.md)
