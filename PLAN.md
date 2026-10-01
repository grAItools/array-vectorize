# Implementation Plan: `vectorizer` — source-to-source array vectorization

Status: **plan only — no code written yet.** Milestone statuses in §10.

## 1. Goal

`vectorize(f)` takes an inspectable **scalar** Python function, parses its source, and
returns a **semantically equivalent function over arrays** that uses only the
[Python Array API standard](https://data-apis.org/array-api/latest/) — so one generated
function runs on NumPy, PyTorch, JAX, CuPy, `array-api-strict`, etc. Unlike
`numpy.vectorize` (a Python-level element loop), the result is genuine vectorized code.
The generated source is inspectable via `inspect.getsource`, a `.source` attribute, and
tracebacks.

**Non-goals (v1):** data-dependent loops (`while`), side effects,
generators/comprehensions, `*args/**kwargs`, mutating statements, class methods. These
are **rejected with precise diagnostics**, never silently miscompiled.

---

## 2. Architecture

```
f (scalar fn)
 └─ _extract:    unwrap decorators → inspect.getsource → dedent → ast.parse
 └─ _validate:   linter pass — collect ALL unsupported nodes → VectorizationError
 └─ _lower:      AST → IR (SSA expression DAG; if-joins as Where merges)
 └─ _optimize:   const-fold, DCE (CSE: simple structural hashing, see §8)
 └─ _codegen:    IR → readable Python source using xp.* only (build ast → unparse)
 └─ _runtime:    compile → exec → linecache registration → callable + .source
 → vectorized callable
```

```
src/vectorizer/
  __init__.py     # vectorize, VectorizationError, get_source
  _extract.py     # source extraction, closure/constant capture          [M1]
  _validate.py    # supported-subset checking + diagnostics              [M1]
  _ir.py          # IR node dataclasses, SSA environment                [M1]
  _lower.py       # AST → IR (the core translator)                      [M1]
  _builtins.py    # scalar-name → Array API mapping tables              [M1]
  _optimize.py    # const-fold / DCE / CSE passes                       [M1]
  _codegen.py     # IR → source text                                    [M1]
  _runtime.py     # compile/exec/linecache, wrapper attributes          [M1]
  _fallback.py    # opt-in np.vectorize-style element loop              [M3]
  _verify.py      # runtime differential-verification helper            [M4]
  _fuzz.py        # grammar fuzzer CLI (python -m vectorizer.fuzz)      [M4]
```

Rationale for an IR (vs direct AST→AST): merging branch-assigned variables, CSE,
readable SSA naming (`x`, `x_1`, `x_2`), and golden-source testing all become
straightforward.

---

## 3. Public API

```python
from vectorizer import vectorize

@vectorize                                   # or vec_f = vectorize(f)
def psi(x):
    if x < 0:
        return 0.0
    return x * math.exp(-x)

psi(x_arr)                                   # any Array API backend
psi.source                                   # generated source (str)
inspect.getsource(psi)                       # also works
vectorize(f, strict=True)                    # default: reject unknowns
vectorize(f, fallback=True)                  # opt-in element-loop fallback  [M3]
vectorize(f, protect_domains=True)           # clamp partial fns in dead lanes [M4]
vectorize(f, verify=inputs_example)          # differential check at gen time [M4]
```

Vectorization happens eagerly at decoration time (fail fast). Signature, defaults,
kwarg names preserved via `__wrapped__`.

---

## 4. Front end: source extraction

- `inspect.unwrap` first (decorated functions); keep original for `__wrapped__`.
  `vectorize(vectorize(f))` must unwrap before extracting, or detect its own marker.
- Reject builtins/C-implemented functions (`inspect.isbuiltin`) with a clear message.
- `inspect.getsource` + `textwrap.dedent` + `ast.parse`; support plain functions and
  lambdas; reject functions defined in REPL/`exec` (no source) with guidance.
- Decorator lines in extracted source are ignored (parse, take `FunctionDef`/`Lambda`).
- `inspect.getclosurevars` + `inspect.signature`:
  - closure/global **scalars** (int/float/bool) frozen as IR literals;
  - names bound to mapped `math` functions (e.g. `from math import sqrt`) resolved by
    object identity against the math table;
  - closure/global **arrays** → hidden trailing parameters with defaults bound at
    generation time (call signature unchanged);
  - anything else unresolvable → reject with hint.

---

## 5. Translation rules (v1 supported subset)

| Scalar construct | Vectorized equivalent |
|---|---|
| `+ - * /`, unary `-` `+` | same operators on arrays |
| `x ** y` | `xp.pow(x, y)` (explicit fn: signals divergence, see §7) |
| `x // y` | `xp.floor_divide(x, y)` |
| `x % y` | `xp.remainder(x, y)` (Python sign semantics) |
| `& \| ^ << >> ~` (ints) | same operators |
| `abs(x)` | `xp.abs(x)` |
| `min(a,b,...)` / `max(a,b,...)` | left-fold `xp.minimum` / `xp.maximum` (variadic) |
| `round(x)` | `xp.round(x)` |
| `round(x, n)` | **open**: `xp.round(x, decimals=n)` if supported by array-api-strict, else reject in M1 — verify at implementation time |
| `int(x)` / `float(x)` / `bool(x)` | `xp.astype(x, xp.int64/float64/bool)` (astype truncates toward zero, matching `int()`) |
| `math.sqrt exp expm1 log log1p log2 log10 sin cos tan asin acos atan atan2 sinh cosh tanh asinh acosh atanh pow floor ceil hypot copysign isnan isinf isfinite` | corresponding `xp.*` |
| `math.trunc(x)` | `xp.astype(x, xp.int64)` (trunc-toward-zero == `int()`) |
| `math.pi/e/tau/inf/nan` | float literals / `xp.inf` / `xp.nan` |
| comparisons `== != < <= > >=` | same, elementwise bool |
| chained `a < b < c` | `xp.logical_and(a < b, b < c)` |
| `and` / `or` / `not` | see §7 D2 (exact value-select lowering) |
| ternary `a if c else b` | `xp.where(coerce(c), a, b)` |
| `if/elif/else` + assignments | nested `xp.where` + SSA variable merging **[M2]** |
| early `return` | merged into result expression **[M2]** |
| assignment / `+=` etc. | SSA rebinding (`x_1`, `x_2`, …) **[M1: straight-line only]** |
| `for i in range(N)` (N constant) | real loop in generated code over arrays **[M3]** |
| call to another pure scalar fn | recursively vectorized, called as `g_vec(...)` **[M3]** |

Statements supported in M1: `Assign` (single `Name` target), `AugAssign` (single
`Name`), exactly one `Return` as the **final** statement, docstring `Expr`, `Pass`.
Everything else — `if`, loops, `global`/`nonlocal`, multiple/tuple assignment,
multiple returns, statements after `return`, annotated assignment — rejected.

Excluded from the math table pending standard confirmation: `cbrt`, `degrees`,
`radians`, reductions (`fsum`, `prod`, `dist`, …), `gcd`, `factorial`, `isclose`,
`ldexp`, `frexp`, `fmod` (verified absent from array-api-strict; not in the
standard — differs from `remainder` on sign, so no safe substitution).

---

## 6. IR design (settled)

Frozen, hashable dataclasses (hashability enables structural CSE):

```python
Literal(value, kind: 'int'|'float'|'bool')   # inf/nan allowed; codegen → xp.inf/xp.nan
Ref(name)                                    # reference to a param or SSA binding
BinOp(op, left, right)                       # add sub mul div pow floordiv mod
                                             # and or xor lshift rshift
UnaryOp(op, operand)                         # neg pos invert not
Compare(op, left, right)                     # eq ne lt le gt ge
Logical(op, parts)                           # and or   (bool operands only)
Where(cond, then, other)
Call(fn, args)                               # fn = Array API name, e.g. 'sqrt'

Binding(name, expr)                          # let-statement
Program(params, bindings, result)            # single return value in M1
```

- No separate `Param` node: parameters are `Ref`s of their own name; first reference
  to `x` emits `x`, reassignment binds `x_1`, etc.
- **SSA naming**: on rebind pick `x_N` with the smallest free N; skip names already
  used by the user (`x` and `x_1` can both be user names). Reserved names
  `{'xp', 'array_namespace'}`: a user variable with a reserved base name is mangled
  with a trailing underscore (`xp` → `xp_`, `xp_1`, …) — only in that rare case.
- **is_bool lattice** (for and/or/ternary lowering, §7 D2): `Compare`, `Not`,
  `Logical`, bool `Literal`, `isnan/isinf/isfinite` calls → bool; everything else →
  non-bool.

---

## 7. Core semantics decisions

- **D1 — Eager branches (dead-lane evaluation)** *(activates in M2)*. `if x > 0:
  y = sqrt(x) else: y = 0` lowers to `xp.where(x > 0, xp.sqrt(x), 0)`: both branches
  evaluate on all lanes; NaN/warnings in dead lanes are discarded by `where`. Final
  values correct; scalar code would raise only on the taken path. `protect_domains`
  [M4] inserts argument clamps for partial functions so dead lanes never leave the
  domain — never changes results, adds ops.
- **D2 — `and`/`or`/`not` exact value-select lowering.** Python `x and y` returns `x`
  if falsy else `y`. Numeric operands: `xp.where(x != 0, y, x)` — **exact**, including
  NaN (`NaN != 0` is True, NaN is truthy). Same for `or` → `xp.where(x != 0, x, y)`.
  When **both** operands are bool-typed (§6 lattice) emit `xp.logical_and/or`
  instead (identical result, prettier source). `not x` → `xp.logical_not(x)` — exact
  for numerics and bools (`logical_not(NaN)` is False; Python `not NaN` is False).
  Chained `a and b and c` lowers left-associatively.
- **D3 — Numeric exceptions become IEEE values.** Division by zero → inf/NaN instead
  of `ZeroDivisionError`; `x % 0` → NaN; int `// 0` → backend-defined (NumPy: 0 +
  warning). `(-8) ** 0.5` → NaN, not complex (Python scalar returns complex).
  Documented divergences, inherent to vectorization.
- **D4 — Bare truthiness** (`if x:`) rejected (ambiguous per-lane) [M2]; suggest
  `if x != 0:`. Ternary with numeric condition is fine: coerced via `!= 0`.
- **D5 — Reject list**: `while`, `break`/`continue`, side-effecting calls
  (`print`, mutation), attribute/item mutation, comprehensions, generators,
  `global`/`nonlocal`, `*args`/`**kwargs`, `try`, `match`, `lambda` inside body,
  walrus, f-strings, subscripts, tuple/list/set/dict literals, tuple returns,
  multiple assignment targets, statements after `return`, no-return functions,
  zero-argument functions, non-literal defaults, str/None/complex literals. The
  validator collects **all** violations at once (linter-style) with file/line/col +
  caret excerpt.
- **D6 — Loops** [M3]: constant-trip `for i in range(...)` only, emitted as a genuine
  loop in the generated source over whole arrays; loop-carried variables become SSA
  phis at the loop head.
- **D7 — Helper calls** [M3]: callee inspectable + pure → recursive `vectorize`,
  memoized by function object. Unknown callable → reject in `strict` (default); with
  `fallback=True`, wrap in an `np.vectorize`-style element loop + `UserWarning`.
- **D8 — Constants/closures** frozen at generation time (§4).
- **D9 — Backend neutrality**: generated code contains **only** `xp.*` calls plus one
  `from array_api_compat import array_namespace` import. Namespace resolved at call
  time:
  ```python
  xp = array_namespace(*[a for a in (x, y) if hasattr(a, "__array_namespace__")])
  ```
  (filters defaulted-out scalar args; at least one argument must be an Array API
  array). A test greps generated sources to enforce zero `np.`/`numpy` leakage.
- **D10 — Reject over miscompile.** Anything not provably translatable within these
  rules raises `VectorizationError`; no best-effort guesses.

---

## 8. Optimizer passes

- **const-fold**: fold BinOp/UnaryOp over Literals. Skip `pow` entirely; skip
  div/mod by literal zero (runtime inf/NaN per D3); skip int results outside int64
  range. Float folding is exact (Python floats are IEEE doubles).
- **DCE**: reachability walk from `result` over `Ref` edges; drop unused bindings.
- **CSE** (simple structural hashing): eligible nodes = `Call`, `Where`; count
  occurrences across bindings+result; nodes seen ≥2 times become temps (`t_1`, …)
  inserted just before the first binding that uses them (all bindings are at function
  scope in M1 — no branches — so hoisting is always safe). Deferred to M2 if time
  runs short; not needed for correctness (chained comparisons merely re-emit plain
  names).

---

## 9. Codegen & inspectability (settled approach)

- **Codegen builds an `ast.Module` and uses `ast.unparse`** (not string templating):
  correct docstring/literal escaping for free, normalized stable formatting.
  Generated module shape:
  ```python
  from array_api_compat import array_namespace


  def f_vec(x, y):
      """<original scalar source, verbatim>"""
      xp = array_namespace(*[a for a in (x, y) if hasattr(a, "__array_namespace__")])
      v_1 = xp.sqrt(x)
      ...
      return v_3
  ```
- `Literal(inf)`/`Literal(nan)` must emit `xp.inf`/`xp.nan` — `ast.unparse` renders
  bare `inf`/`nan` identifiers, which are **invalid Python**.
- Operator forms: `+ - * / & | ^ << >> ~ -` stay operators; `pow`, `floor_divide`,
  `remainder`, `logical_*`, `where`, `astype`, all math fns → explicit `xp.*` calls.
- **Golden comparison via `ast.dump` equality** (parse golden file, parse generated
  source, compare trees) — immune to `unparse` formatting drift across Python
  versions; golden `.py` files remain human-reviewable.
- Compile as `compile(src, "<vectorizer:f>", "exec")`; register source in
  `linecache.cache` as `(len(src), None, src.splitlines(True), filename)` — entries
  with `mtime=None` survive `linecache.checkcache`, so:
  - `inspect.getsource(vec_f)` / `getsourcelines` work,
  - **tracebacks render the real generated lines**,
  - `help()` works; `__name__/__qualname__/__doc__/__wrapped__` set on the wrapper.
- The returned object is the compiled function itself (not a wrapper object), with
  `.source` attribute and `__wrapped__` → original. `inspect.unwrap` follows
  `__wrapped__`, so `inspect.getsource` must be checked/tested to return the
  **generated** source, not the original's.
  **Implementation note (deviation):** CPython's `inspect.getsourcelines`
  unwraps `__wrapped__` unconditionally, so setting it would hide the generated
  source. Instead the runtime sets `__signature__` (signature/defaults/kwarg
  names preserved) and a `_vectorized_original` marker (plan §4's "detect its
  own marker" option) used for `vectorize(vectorize(f))` idempotence.

---

## 10. Milestones

| M | Scope | Acceptance criteria | Status |
|---|---|---|---|
| **M0** | Scaffolding: `src/` layout, `pyproject.toml`, ruff + mypy(strict) + pytest + coverage + pre-commit, GitHub Actions (py 3.10–3.13), empty package | `make check` green in CI | **done** |
| **M1** | MVP pipeline: extraction, validation, expressions (§5 minus control flow) + straight-line assignments, codegen, `linecache` inspectability, golden harness, differential (Hypothesis) on NumPy | 20 golden cases exact-match; `inspect.getsource` works; differential green | **done** (34 goldens) |
| **M2** | Statement semantics: `if/elif/else` merges, early returns, full reject diagnostics polish | `relu/clamp/piecewise/psi`-class functions pass behavioral + differential; error tests complete | **done** |
| **M3** | Composition: constant-trip loops, recursive helper vectorization, closures/defaults, `fallback` mode | composition tests complete; backend matrix green | not started |
| **M4** | Safety & polish: `protect_domains`, `verify=` option, README with semantics table + divergence docs, examples, fuzzer CLI | divergence tests complete; docs reviewed | not started |
| **M5** (stretch) | on-disk source cache keyed by (source hash, py version), `scipy.special` maps, fuzzer hardening | nightly fuzz 24h clean | not started |

---

## 11. Testing strategy (extensive)

- **T1 — Unit tests per pass** (table-driven): extraction edge cases; validator
  accepts/rejects per construct; lowering shape; codegen trees.
- **T2 — Golden source snapshots**: corpus of ~60 input functions covering *every*
  rule in §5. Exact-match via `ast.dump` equality against `tests/golden/*.py`;
  `pytest --update-golden` regenerates; diffs human-reviewable.
- **T3 — Behavioral tests** (NumPy): generated function vs hand-computed arrays.
- **T4 — Differential testing vs the scalar original** (Hypothesis): random scalar
  inputs with edge-value strategy (`0, ±1, -0.0, subnormals, ±inf, NaN, int
  extremes`), dtypes `float32/64, int32/64`; `equal_nan=True`, tight tolerances.
  CI profile: `derandomize=True`, ~100 cases; nightly: 10k cases.
- **T5 — Second oracle**: differential vs `np.vectorize(f)`.
- **T6 — Backend matrix** (parametrized, skip-if-missing): `numpy` (reference),
  `array-api-strict` (conformance + promotion + no-leakage), `torch` (cpu), `jax`,
  `cupy` (optional).
- **T7 — Error/diagnostic tests**: every reject-list construct → `VectorizationError`
  with correct line/col; message snapshots.
- **T8 — Documented-divergence tests**: dead-lane NaN never leaks into results
  (M2+); div-by-zero → inf; `%`/`//` sign semantics on negatives; `round` half-even;
  eager `and`/`or` correctness incl. NaN lanes; `(-8)**0.5` → NaN.
- **T9 — Inspectability tests**: `inspect.getsource`/`getsourcelines` return
  generated source (not original via `__wrapped__`); `linecache` registration
  survives `checkcache()`; traceback line rendering; `.source`; docstring contents;
  signature/defaults/kwargs forwarding; `vectorize(vectorize(f))` idempotence.
- **T10 — Composition tests** [M3]: constant-trip loops, recursive helper
  vectorization + memoization, closures, defaults, annotations, lambdas.
- **T11 — Performance smoke** (slow-marked): ≥50× faster than `np.vectorize` at 10⁶
  elements.
- **T12 — Grammar fuzzer** [M4]: `python -m vectorizer.fuzz --seconds 60`, random
  programs from the supported grammar, differential-checked, reproducible seeds,
  nightly CI.
- **Coverage & hygiene**: ≥95% branch coverage on translator modules; every bugfix
  lands with a regression test; optional `mutmut`.

---

## 12. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Eager branches cause warnings/NaN noise | `protect_domains` clamping [M4]; NumPy-only `errstate` escape hatch; documented |
| Backend FP differences (libm variance) | tolerances in differential tests; per-backend tolerances if needed |
| Array API promotion surprises | `array-api-strict` in the matrix from M1 |
| AST/`linecache`/`unparse` behavior varies across CPython versions | CI matrix 3.10–3.13; T9 asserts mechanics per version; golden via `ast.dump` equality |
| `xp.round` `decimals` kwarg may not be standard | verify against array-api-strict at implementation time; reject `round(x, n)` in M1 if unsupported |
| Scope creep toward a full compiler | supported subset enforced by validator; reject list is the contract |
| Golden files churn on formatting | `ast.unparse` normalized output + tree-equality comparison; updates via flag only |

---

## 13. Tooling & CI

`pytest`, `hypothesis`, `ruff`, `mypy`, `coverage[toml]`, `array-api-strict`,
`array-api-compat`, `pre-commit`; optional extras `torch`, `jax`, `cupy`. CI:
lint + type + tests + coverage gate on PRs; nightly: fuzz campaign + 10k-case
Hypothesis profile. Python ≥ 3.10 (relies on `ast.unparse`, stable `linecache`
semantics). Dev commands via `make` / `make check` (lint + type + test + coverage).
