---
icon: lucide/function-square
---

# array-vectorize

`vectorize(f)` compiles an inspectable **scalar** Python function into a
**semantically equivalent function over arrays** that uses only the
[Python Array API standard](https://data-apis.org/array-api/latest/) — one
generated function runs on NumPy, PyTorch, JAX, CuPy, `array-api-strict`,
and any other standard-compliant backend.

Unlike `numpy.vectorize` (a Python-level element loop), the result is
genuine vectorized code, and the generated source is readable and
inspectable.

=== "Scalar input"

    ```python
    @vectorize
    def psi(x):
        if x < 0:
            return 0.0
        return x * math.exp(-x)
    ```

=== "Generated source (`psi.source`)"

    ```python
    from array_api_compat import array_namespace

    def psi_vec(x):
        """
        @vectorize
        def psi(x):
            if x < 0:
                return 0.0
            return x * math.exp(-x)
        """
        xp = array_namespace(x)
        return xp.where(x < 0, 0.0, x * xp.exp(-x))
    ```

```python
x = np.asarray([-2.0, -0.5, 0.0, 1.0, 3.0])
psi(x)   # array([0.        , 0.        , 0.        , 0.36787944, 0.14936121])
```

The same `psi` runs unchanged on torch tensors, JAX arrays, CuPy arrays,
or `array_api_strict` arrays.

## Installation

```bash
pip install "array-vectorize @ git+https://github.com/grAItools/array-vectorize.git"
```

Requires Python >= 3.12. The only runtime dependency is `array-api-compat`.
The package is not on PyPI yet; once published, plain
`pip install array-vectorize` will work. For development from a checkout,
see [Development](development.md) (`make install`, then `make check`).

## Highlights

- **Reject over miscompile** — anything outside the
  [supported subset](supported-subset.md) raises `VectorizationError` with
  a precise diagnostic (line/column + caret), never silent wrong results.
- **Exact integer semantics** — `min`/`max`, boolean arithmetic, and
  uint64/int64 mixed arithmetic go through runtime helpers that produce
  exact results whenever a single dtype can hold them
  ([semantics](semantics.md)).
- **Inspectable output** — `vec.source` (and `inspect.getsource`) show the
  generated function; tracebacks map to real lines.
- **Differentially tested** — Hypothesis differential tests, 45 golden
  source snapshots, a grammar fuzzer, and a 26-round adversarial review
  with every finding regression-tested.
