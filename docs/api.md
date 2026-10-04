# Public API

## `vectorize`

```python
vectorize(f, *, strict=True, fallback=False, protect_domains=False,
          verify=None)
```

Compile a scalar function into an Array API function.

| Parameter | Default | Meaning |
|---|---|---|
| `strict` | `True` | raise `VectorizationError` on unsupported constructs; `False` falls back to an element loop |
| `fallback` | `False` | emit a `UserWarning` when the element-loop fallback is used |
| `protect_domains` | `False` | clamp partial-function arguments in provably dead lanes |
| `verify` | `None` | example arguments for a differential check at generation time |

Vectorization happens eagerly at decoration time. Signature, defaults,
and kwarg names are preserved. `vectorize(vectorize(f))` is idempotent.

## Vectorized functions

| Attribute | Meaning |
|---|---|
| `vec.source` | the generated Python source (`str`) |
| `inspect.getsource(vec)` | works; tracebacks show real generated lines |
| `get_source(vec)` | helper that raises `VectorizationError` for non-vectorized input |

## Errors

`VectorizationError` — raised for unsupported constructs, with all
violations collected at once and precise line/column positions:

```
VectorizationError: <file>:3:5: 'while' loops are not supported
                   (supported: 'for i in range(N)' with constant N)
```

## CLI

```bash
python -m vectorizer.fuzz --seconds 60 --seed 0
```

Runs the grammar fuzzer (differential checks of random scalar programs).
Reproducible with `--seed`.
