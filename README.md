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
#     @vectorize
#     def psi(x):
#         if x < 0:
#             return 0.0
#         return x * math.exp(-x)
#     """
#     xp = array_namespace(x)
#     return xp.where(x < 0, 0.0, x * xp.exp(-x))
```

The same `psi` runs unchanged on torch tensors, JAX arrays, CuPy arrays, or
`array_api_strict` arrays.

## Installation

```bash
pip install "array-vectorize @ git+https://github.com/grAItools/array-vectorize.git"
```

Requires Python >= 3.12. The only runtime dependency is `array-api-compat`.
The package is not on PyPI yet; once published, plain
`pip install array-vectorize` will work. From a checkout (development):
`pip install -e ".[dev]"` then `make check` (lint + type + test + coverage).

## Public API

```python
vectorize(f)                        # strict vectorization (default)
vectorize(f, strict=False)          # fall back to an element loop on failure
vectorize(f, fallback=True)         # same, with an explicit UserWarning
vectorize(f, protect_domains=True)  # clamp partial functions in dead lanes
vectorize(f, verify=example_args)   # differential check at generation time
vectorize(f, namespace=np)          # pin the backend (all-scalar calls work)

vec.source                          # generated source (str)
inspect.getsource(vec)              # also works; tracebacks show real lines
get_source(vec)                     # helper that raises for non-vectorized input
vec.with_namespace(xps)             # new callable pinned to another backend
VectorizationError                  # raised on unsupported constructs
```

- Vectorization happens eagerly at decoration time (fail fast).
- Signature, defaults, and kwarg names are preserved.
- `vectorize(vectorize(f))` is idempotent.
- Namespace pinning: `vectorize(f, namespace=xp)` binds `xp` directly at
  decoration time (never extracted from arguments, so all-scalar calls
  work); `vec.with_namespace(xp)` returns a new pinned callable.
- `python -m array_vectorize.fuzz --seconds 60` runs the grammar fuzzer
  (reproducible with `--seed`).

## Supported subset

Anything outside this subset is **rejected with a precise diagnostic**
(line/column + caret) — never silently miscompiled.

| Scalar construct | Vectorized equivalent |
|---|---|
| `+ - * /` (and unary `- +`) | same operators |
| `x ** y` | `xp.pow(x, y)` |
| `x // y`, `x % y` | `xp.floor_divide`, `xp.remainder` (Python sign semantics) |
| `& \| ^ << >> ~` | same operators (integer inputs) |
| `abs`, `round`, `min`, `max` | `xp.abs`, `xp.round`, left-folded `xp.minimum`/`xp.maximum` |
| `int(x)`, `float(x)`, `bool(x)` | `xp.astype(x, xp.int64/float64/bool)` (truncation toward zero) |
| `math.sqrt exp expm1 log log1p log2 log10 sin cos tan asin acos atan atan2 sinh cosh tanh asinh acosh atanh pow floor ceil hypot copysign isnan isinf isfinite` | corresponding `xp.*` |
| `math.trunc(x)` | `xp.astype(x, xp.int64)` |
| `math.pi/e/tau/inf/nan` | literals (`xp.inf` / `xp.nan` for the last two) |
| comparisons, chains (`a < b < c`) | elementwise; chains fold with `xp.logical_and` |
| `and` / `or` / `not` | exact value-select lowering (`xp.where(x != 0, y, x)`); `xp.logical_*` when both operands are boolean |
| ternary `a if c else b` | `xp.where(coerce(c), a, b)` |
| `if/elif/else`, early `return` | nested `xp.where` merges |
| assignment, `+=` etc. | SSA rebinding (`x`, `x_1`, `x_2`, ...) |
| `for i in range(N)` (N constant) | a real loop in the generated source; loop-carried variables become explicit phis |
| call to another pure scalar function | recursively vectorized, memoized by function object |
| closure/global scalars | frozen as constants at generation time |
| closure/global arrays | hidden keyword parameters with bound defaults |

Not supported (rejected): `while`, `break`/`continue`, `return` inside loop
bodies, side effects, comprehensions/generators, `*args`/`**kwargs`,
`try`, `match`, walrus, f-strings, subscripts, tuple/list/set/dict literals,
tuple returns, `global`/`nonlocal`, statements after `return`, annotated
assignments, non-literal defaults, `str`/`None`/`complex` literals, bare
`if x:` truthiness (write `if x != 0:`), recursive helper calls, and
`round(x, n)` (`decimals=` is not in the Array API standard).

## Semantics decisions

- **Eager branches (dead lanes).** `if c: y = f(x) else: y = g(x)` lowers to
  `xp.where(c, f(x), g(x))`: both branches evaluate on all lanes. Values are
  correct; NaN/warnings from dead lanes are discarded by `where`.
  `protect_domains=True` clamps partial-function arguments (`sqrt`, `log`,
  `asin`, ...) on lanes that are provably dead (discarded by their enclosing
  where-select) and out of domain — live lanes are never touched (subnormals
  and signed zeros stay exact), warnings on dead lanes disappear. Loop bodies
  and cross-function (helper) dead lanes are not tracked (conservative).
- **`and`/`or`/`not` are exact.** Numeric operands lower to value-selects
  (`x and y` → `xp.where(x != 0, y, x)`), including NaN truthiness
  (`not nan` is `False` in both Python and `xp.logical_not`). When both
  operands are boolean, `xp.logical_and/or` is emitted instead.
- **Reject over miscompile.** Anything not provably translatable raises
  `VectorizationError` with all violations collected at once.

## Documented divergences from scalar Python

Vectorization is lane-parallel, so a few scalar behaviors become IEEE
values. These are deliberate, tested (T8), and inherent:

| Scalar behavior | Vectorized behavior |
|---|---|
| `ZeroDivisionError` from `1/x` | `inf`/`-inf` (IEEE) |
| `ZeroDivisionError` from `x % 0` | `NaN` |
| `(-8) ** 0.5` returns `complex` | `NaN` |
| `math` domain errors (`sqrt(-1)`) | `NaN` (plus a backend warning) |
| integer overflow in `+`/`*` folds | folding skipped; backend behavior |
| `min(a, nan)` returns `a` | `xp.minimum` propagates `NaN` |
| `round(-0.0)` returns int `0` (sign dropped) | `xp.round` keeps `-0.0` |
| branch-taken type kept (`0 if c else False`) | `xp.where` promotes branch dtypes |
| bool arithmetic is integer (`True + True == 2`) | exact: bool operands are cast to float64 (strict backends reject bool arithmetic) |
| scalar exceptions inside `verify=` | expected `NaN` lanes |
| mixed int/float `min`/`max` siblings | strict backends require matching array dtypes; literals adopt a provably-int sibling's dtype, else float64 |
| uint64 + signed arrays with mixed-sign per-lane results | float64 approximation (no single dtype holds both operand ranges) |

## Development

```bash
make check       # ruff + mypy (strict) + pytest + coverage (gate: 95%)
make fmt         # format + autofix
make fuzz        # grammar fuzzer on the CI seeds
make bench       # pytest-benchmark suite (timings + dispatch gates)
make docs        # build the docs site (Zensical, docs/)
make notebook    # open the marimo example notebooks (examples/notebooks/)
pytest --update-golden   # regenerate golden source snapshots
python -m array_vectorize.fuzz --seconds 60 --seed 0    # grammar fuzzer
```

- Golden snapshots (`tests/golden/cases/`) compare via `ast.dump` equality,
  so formatting drift never breaks them.
- Differential tests (Hypothesis) compare generated functions against the
  scalar originals on edge values (`0, ±1, subnormals, ±inf, NaN`).
- The full docs live in `docs/` (build with `make docs`); interactive
  examples live in `examples/notebooks/` (marimo).
- The docs site is tested and deployed to
  <https://grAItools.github.io/array-vectorize/> automatically on every
  push to `main`.
- See `PLAN.md` for the full design document and milestone history.
