# Public API

## `vectorize`

```python
vectorize(f, *, strict=True, fallback=False, protect_domains=False,
          verify=None, namespace=None)
```

Compile a scalar function into an Array API function.

| Parameter | Default | Meaning |
|---|---|---|
| `strict` | `True` | raise `VectorizationError` on unsupported constructs; `False` falls back to an element loop |
| `fallback` | `False` | emit a `UserWarning` when the element-loop fallback is used |
| `protect_domains` | `False` | clamp partial-function arguments in provably dead lanes |
| `verify` | `None` | example arguments for a differential check at generation time |
| `namespace` | `None` | pin the array namespace at decoration time (see below) |

Vectorization happens eagerly at decoration time. Signature, defaults,
and kwarg names are preserved. `vectorize(vectorize(f))` is idempotent.

### Namespace pinning

`vectorize(f, namespace=np)` pins the array namespace: the generated code
binds `xp` directly to it and never extracts the namespace from its
arguments, so **all-scalar calls become legal**. Passing arguments
compatible with the pinned namespace is the user's responsibility; an
invalid namespace raises `TypeError` (a usage error, not a
`VectorizationError`).

The pinned namespace must itself implement the Array API. NumPy's main
namespace does from NumPy 2.0; on NumPy 1.x, pin
`array_api_compat.numpy` instead (unpinned functions work on NumPy 1.x
unchanged, because `array_namespace` already returns that wrapper).

Pinned generated source has no `array_namespace` import — paste-ready
code must pass the hidden keyword-only parameter (`fn(..., _namespace=xp)`)
or rebind `xp` by hand. `vec.with_namespace(xp)` (below) creates variants
without touching the original.

## Vectorized functions

| Attribute | Meaning |
|---|---|
| `vec.source` | the generated Python source (`str`) |
| `inspect.getsource(vec)` | works; tracebacks show real generated lines |
| `get_source(vec)` | helper that raises `VectorizationError` for non-vectorized input |
| `vec.__doc__` | the original's docstring with a `(array-vectorized)` summary prefix; the scalar source in a `Notes:` section (Google style, or NumPy style when the original's docstring uses NumPy section headers) |
| `vec.with_namespace(xp)` | returns a NEW callable pinned to `xp` (strict results and fallback wrappers) |
| `f.__array_vectorized__` | on the scalar original: its canonical vectorization |

The docstring's `Notes:` section embeds the scalar source verbatim after a
fixed lead-in line, so `help(vec)` shows documentation first and the compiled
function's provenance below. When that round trip cannot be guaranteed (the
lead-in quoted inside the original's own docs, tab-indented source), the
legacy source-only docstring is emitted instead.

Canonical calls — `vectorize(f)` without `protect_domains` or `namespace` —
are memoized per scalar original: repeated calls (including
`vectorize(vectorize(f))`) return the same object, `verify=` still runs on
every call, and `f.__array_vectorized__` points at it. The first
compilation wins: closure scalars are frozen at that point (design D8), so
re-decorating after reassigning a global that `f` reads does not re-capture
it — the same semantics `with_namespace` pins have always had. Option
variants (protected, pinned) never take over the back-reference marker;
fallback wrappers set it but are rebuilt (and re-warn) on every call. Only
plain functions carry the marker — builtins and C callables cannot.

`with_namespace` never mutates the original, preserves `.source`,
`.__signature__`, and `._vectorized_original` (the scalar original), is
memoized (identical pins return the same object), and chains
(`vec.with_namespace(a).with_namespace(b)`). Variants do not re-run
`verify=` — the body code is unchanged, only the `xp` binding differs.

## Errors

`VectorizationError` — raised for unsupported constructs, with all
violations collected at once and precise line/column positions:

```
VectorizationError: <file>:3:5: 'while' loops are not supported
                   (supported: 'for i in range(N)' with constant N)
```

## CLI

```bash
python -m array_vectorize.fuzz --seconds 60 --seed 0
```

Runs the grammar fuzzer (differential checks of random scalar programs).
Reproducible with `--seed`.

| Option | Meaning |
|---|---|
| `--seconds` | time budget in seconds (default: 60) |
| `--seed` | random seed for a reproducible run |
| `--cases` | maximum cases; overrides the time budget |
| `--help` | show usage and exit |
