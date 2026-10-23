# array-vectorize — architecture, implementation, and quality review

- **Date:** 2026-10-23
- **Revision reviewed:** `0e4bb6b` (branch `review-project-defects-enhancements`)
- **Scope:** full source tree (41 modules, 5,771 lines), documentation, tests,
  build/CI/release tooling, agent layout.
- **Verification:** the project gate was run end-to-end, then targeted
  reproduction scripts were written for untested semantic edges.

| Command | Result |
|---|---|
| `make check` | pass — lint, four type checkers, `1064 passed, 97 skipped`, branch coverage **96%** |
| `make docs` | pass — zensical `--strict`, “No issues found” |
| `make smoke` | pass — smoke script, `examples/demo.py`, all marimo notebooks |

Reproduction artifacts live in [`reviews/artifacts/`](artifacts/). They encode
the *contract* checks, so they currently report defects and exit nonzero; each
becomes a regression test once the corresponding issue is fixed.

---

## Executive summary

The architecture is clean and the runtime helpers are carefully reasoned, but
the compiler's central exactness guarantee is conditional on **statically
provable operand kinds**. Function parameters have no compile-time dtype, so
the most common call pattern — `def f(x, y): return x + y` with bool, uint64,
or mixed integer arrays — silently violates the contract on NumPy and raises
on `array-api-strict`. Control-flow merges (`if`, ternary, early return) have
the same hole. There are also silent miscompiles for decorated functions,
`_vec_arith` add/mul with negative literals, `range`/builtin shadowing, a
linecache collision that corrupts `inspect.getsource`, and `verify=` failures
and blind spots.

The green gate is compatible with all of this because the suite is
dtype-narrow: line coverage is 96%, and the faulty `_vec_arith` branch is
covered, but only with inputs whose results stay non-negative.

| # | Finding | Severity | Status |
|---|---|---|---|
| F1 | Exactness fails for call-time bool/uint64/mixed dtypes; merges promote uint64/int64 to float64; `_vec_arith` add/mul wrap negative results | **P0** | verified |
| F2 | `inspect.unwrap` compiles the wrong function for decorated targets | **P0** | verified |
| F3 | Scalar parameters and loop indices are passed raw to `xp.*` / `xp.where`, breaking strict backends | **P1** | verified |
| F4 | `range` and builtin shadowing are ignored | **P1** | verified |
| F5 | Generated filename collision corrupts `inspect.getsource`/tracebacks; caches never evict | **P1** | verified |
| F6 | `verify=` rejects documented divergences, misses large-int errors, breaks on frozen globals | **P1** | verified |
| F7 | Captured arrays / NumPy scalars lock a function to one backend | **P1** | verified |
| F8 | Literal/int64 invariant escapes; int64 minimum unusable | **P1** | verified |
| F9 | `floor/ceil`, `abs`, `round`, casts, negative-exponent `pow`, zero-divisor integers are not exact/consistent | **P1** | verified |
| F10 | Diagnostics contradict docs: no filename, snippet-relative lines, byte-offset carets | **P2** | verified |
| F11 | Cache/thread/fallback API issues | **P2** | verified/inferred |
| F12 | Optimizer and runtime-helper performance opportunities | **P2/P3** | verified/inferred |
| F13 | Documentation inaccuracies | **P2** | verified |
| F14 | Testing gaps that allowed F1–F9 to survive | **P2** | verified |

---

## What is genuinely strong

Worth preserving while fixing the above:

- **Stage separation and layering** (`runtime` stdlib leaf, `ir`
  dependency-free, `pipeline` vs `api`) is enforced by
  `tests/unit/test_layering.py`, not just documented.
- **The exactness helpers themselves** (`runtime/arith.py`, `minmax.py`,
  `dtype.py`) are the best part of the codebase: result-aware subtraction,
  uint64 modular paths, guard/sticky-bit integer division, bool normalization,
  never-winning-bound clamps. They are documented and defensively written.
- **Reject-over-miscompile tooling**: collected diagnostics, `ast.unparse`
  codegen, `linecache` registration, docstring round-trip self-check, iterated
  loop kind fixed point.
- **Project engineering**: lockfile + floor verification, four blocking type
  checkers, cleanporter, canonical headers, golden snapshots compared via
  `ast.dump`, docs consistency tests, release metadata/artifact validation,
  Linux/Windows/Python-matrix CI, fuzz seeds.
- **Documentation is unusually honest** about design decisions D1–D10; the
  problems below are mostly places where implementation drifted from that
  documentation.

---

## F1 — The exactness contract only holds when dtypes are provable at compile time (P0)

The lowering decides whether to call `_vec_arith` / `_vec_minmax` from
`_numeric_kind` (`src/array_vectorize/lower/expressions.py:351-407`).
Parameters are marked only `maybe_scalar`, never given a kind, so risk
operations on parameters compile to bare operators or `xp.where`. The docs
claim the opposite:

- `README.md`: “Exact semantics — … boolean arithmetic, and mixed
  uint64/int64 arithmetic match scalar Python wherever a single dtype can hold
  the result.”
- `docs/semantics.md`: “bool arithmetic is integer (`True + True == 2`) …
  exact: bool operands are cast to float64”.
- `docs/supported-subset.md` lists `+ - * /`, bitwise ops, and `min/max`
  without a dtype contract.

### F1.1 Call-time dtype holes

All with `def f(x, y): return <expr>` and `vectorize(f)`:

| Operation | Inputs | Scalar Python | Generated result |
|---|---|---|---|
| `x + y` | `bool[True,True,False]`, `bool[True,False,True]` | `[2, 1, 1]` | NumPy: `[True, True, True]`; strict: `TypeError` |
| `x * y` | bool | `[1, 0, 0]` | NumPy: bool array |
| `x - y`, `-x`, `+x` | bool | ints | NumPy: `TypeError` / `UFuncTypeError` |
| `~x` | bool | `[-2, -1]` | `[False, True]` (logical not) on **both** NumPy and strict |
| `abs(x) + abs(x)` | bool | `[2, 0]` | NumPy: `[True, False]` — **wrong value** |
| `x + y` | uint64 `2**64-1` + int64 `1` | `18446744073709551616` | float64 (lossy) |
| `x - y` | uint64 `2**64-1` - int64 `1` | `18446744073709551614` | float64 (off by 2) |
| `x + -1` / `x * -2` | uint64 | exact | `OverflowError` (negative literal boxed against uint64) |
| `x // y` / `x % y` | uint64 vs int64 | exact | float64; `(2**64-1) % 3` → `1.0` instead of `0` |
| `x & y` | uint64 vs int64 | exact | `TypeError: ufunc not supported` |
| `abs(x)` | int64 `-2**63` | `9223372036854775808` | `-9223372036854775808` (overflow) |

### F1.2 Merges and phis

The compiler's central feature — `if`/ternary/early-return merging — uses
`xp.where` (`lower/env.py:150-180`, `:200`) with no dtype dispatch:

```python
def f(c, x, y):
    if c > 0:
        r = x
    else:
        r = y
    return r


# x = uint64[2**63 + 1], y = int64[-5]
# generated: xp.where(c > 0, x, y) -> float64 [9.22337204e18, -5.0]
# expected: [9223372036854775809, -5]
```

Identical for ternaries, early returns, and `and`/`or` value-selects.
`ir.is_bool` does not see `BinOp`/`Where`/`FuncCall` (`ir/nodes.py:188-200`),
so bool-ness is also lost through intermediate expressions.

### F1.3 `_vec_arith` add/mul wrap negative results

The `others_intish` branch (`runtime/arith.py:275-277`) does modular uint64
for `add`/`mul`, unlike the result-aware `sub`:

```python
def f(x):
    a = min(x, True)
    return a + -1


vectorize(f)(np.array([0], dtype=np.uint64))
# -> [18446744073709551615] uint64   expected -1


def g(x):
    a = min(x, True)
    return a * -1


vectorize(g)(np.array([1], dtype=np.uint64))
# -> [18446744073709551615] uint64   expected -1
```

This branch is exercised by the suite
(`test_uint64_negative_literal_arithmetic_exact`,
`tests/behavior/test_arithmetic_dtypes.py:605`), but only with a positive
result (`max(x, True) + -2` with `2**63+3`).

### F1.4 Root cause and proposals

Static kind inference is treated as a proof of runtime dtype, but it is only a
proof for literals, casts, comparisons, and a few helper results. Choose one
contract and enforce it:

1. **Runtime dispatch for every risk op** (recommended): route `+ - *`, unary
   `-`/`+`, `~`, shifts, `pow`, and floor/mod through `_vec_arith` (or small
   fast-path wrappers that check `dtype` and fall back to the native operator)
   whenever either operand is not provably float. Add a `_vec_where` that
   implements the same per-lane dtype lattice for merges/phís — this alone
   closes if/ternary/early-return/`and`/`or`.
   **This requires fixing F1.3 first or simultaneously**:
   - add with negative scalar `n`: normalize operand order and call
     `_u64_sub_exact(xp, array, -n)`.
   - mul with negative scalar `n` (`k = -n`): guard
     `all(array <= 2**63 // k)`; when true, compute the uint64 product
     (guaranteed ≤ `2**63`, no wrap) and negate via
     `_u64_sub_exact(xp, 0, product)` (exact at the int64-min edge). When
     false, compute the documented float64 fallback from the original array,
     never from a wrapped product. Handles `a * -k` and `-k * a`.
2. **Declared dtype contract**: `vectorize(f, dtypes={...})` or
   `assume_float=True`, which specializes codegen or installs a call-time
   guard that rejects unsupported dtypes with a clear `TypeError`.
3. At minimum, **detect and reject** bool/uint64 inputs for unfixed operations
   rather than producing backend-dependent results.

Also decide and document division's dtype policy: `x / y` on `float32` returns
`float64` (helper generic path, `runtime/arith.py:231-235`), while
`+ - * // %` preserve `float32`.

---

## F2 — Decorated functions are compiled from the inner function (P0)

`resolve_original` calls `inspect.unwrap(func)` unconditionally
(`frontend/extract.py:36-46`). Any decorator built with `functools.wraps` is
stripped, even when it changes behavior:

```python
def deco(fn):
    @functools.wraps(fn)
    def wrapper(x):
        return fn(x) + 100.0

    return wrapper


@deco
def f(x):
    return x * 2.0


# scalar f(3.0) == 106.0
# vectorize(f)(array([3.0])) == [6.0]   <-- silent D10 violation
```

The wrapper's source is ignored; `vec._vectorized_original` points at the
wrapper while the body comes from `fn`, and
`f.__array_vectorized__` is set on the unwrapped inner function, not on the
object the user decorated.

**Proposal.** Follow only this library's own markers (`_vectorized_original`
and the fallback wrapper's marker). Do not unwrap arbitrary `__wrapped__`.
Either compile the wrapper's source (usually rejected with a clear message) or
reject “decorated functions are not supported; pass the undecorated function”
unless the decorator opted in. Add a regression test with a behavior-changing
`functools.wraps` decorator.

---

## F3 — Raw scalar arguments and loop indices break backend neutrality (P1)

“Maybe scalar” tracking is used only in `_sanitize_xp_arg` for math calls,
casts, and builtins (`lower/sanitize.py:65-99`). It is **not** used by:

- the `xp.pow`/`floor_divide`/`remainder` lowering;
- `xp.where` conditions;
- `xp.logical_not`/`logical_and`/`logical_or`.

Reproduced with `array-api-strict`:

```python
def f(x):
    s = 0.0
    for i in range(3):
        s = s + (i // 2) + (i % 2) + (i**2)
    return s + x


# generated: xp.floor_divide(i, 2) + xp.remainder(i, 2) + xp.pow(i, 2)
# strict: TypeError: Two scalars not allowed


def f(x, flag):
    if flag > 0:
        return x
    return -x


# generated: xp.where(flag > 0, x, -x)
# strict: TypeError: `condition` must be an Array; got <class 'bool'>


def f(x):
    return not x


# generated: xp.logical_not(x)
# strict: TypeError: Only boolean dtypes are allowed in logical_not; got float64
```

NumPy tolerates all three, so the bugs are invisible in the default backend.
The stale comments in `lower/expressions.py:209` and
`tests/backends/test_array_api_strict.py:31-39` describe a parameter
normalization preamble that **does not exist** in generated source.

**Proposals.**

- Restore a generated preamble that normalizes parameters
  (`x = xp.asarray(x)`), or wrap arguments at every `xp.*`/`xp.where`/
  `logical_*` site using the existing `_possibly_scalar` analysis.
- Lower `not numeric` to `x == 0` (preserves Python/NaN truthiness and is
  strict-safe) instead of `logical_not(x)`.
- Add strict-backend tests for scalar flags, loop-index arithmetic, and `not`;
  fix the misleading “preamble” test name or make it assert the real
  normalization.

---

## F4 — Name hygiene: `range` and builtin shadowing (P1)

`validate_for` and `lower_for` treat any `for i in range(...)` as the builtin
(`frontend/validate.py:185-202`, `lower/loops.py:30-38`) without resolving
`range` against locals, parameters, or closures.

Verified:

- `def f(x, range): for i in range(3): ...` compiles; calling it raises
  `TypeError: 'int' object is not callable`.
- A live local `range = 3` compiles and raises similarly (DCE hides it only
  when the binding is dead).
- A global function `range` that expands the trip count compiles silently
  wrong: scalar loops 3 times, vectorized loops 2.

Builtins have a related hole: `_lower_call` checks locals
(`definite`/`maybe`/`assigned_names`) but not `closure_scalars`/
`closure_arrays` before dispatching to `BUILTIN_UNARY/FOLDS/CASTS`
(`lower/expressions.py:145-202`). `abs = 5; def f(x): return abs(x)` and the
same for `int` compile using the builtin, while the scalar original raises
`TypeError`.

Separately, parameters named `array_namespace` or `hasattr` are mangled
(`ir/ssa.py:19`), but `__signature__` still advertises the original name
(`emit.py:78`):

```python
def f(array_namespace):
    return array_namespace + 1


inspect.signature(v)  # (array_namespace)
v(array_namespace=array([1.0]))  # TypeError: unexpected keyword argument
```

**Proposals.** Resolve every builtin-looking name against visible bindings
first and reject shadowed builtins with a positioned error (or route to a
supported helper). For reserved names, reject parameter names that would
require renaming, or emit a forwarding wrapper — never advertise a signature
the function does not accept.

---

## F5 — Generated filenames collide; inspectability and caches leak (P1)

`emit.compile_vectorized` uses `filename = f"<array_vectorize:{name}>"`
(`emit.py:46`) and overwrites `linecache` (`:66`).

Verified:

- Two different modules each defining `f`: `inspect.getsource(v0)` returns
  `v1`'s generated source.
- The same function compiled canonically and with `protect_domains=True`:
  `inspect.getsource(v1)` returns the protected variant's source while
  `v1.source` is correct.
- Tracebacks show the wrong lines for the older function.

`tests/behavior/test_inspectability.py:58-66` asserts the exact colliding
filename, locking the bug in.

Additionally, `_HELPER_CACHE`, `_CANONICAL_CACHE`, `_PIN_CACHE`
(`api.py:39-49`) are strong, unbounded dictionaries; `linecache` entries are
never removed. Long-running services and the fuzzer (one new function object
per case) grow without bound.

**Proposals.** Make the generated filename unique and readable
(`<array_vectorize:module.qualname#id>` or a counter stored on the function),
keep `vec.source` as the canonical text, and update the inspectability test to
assert uniqueness across same-named functions and option variants. Add
`clear_caches()` (and optionally weak references) plus documentation.

---

## F6 — `verify=` models divergences incorrectly and has a comparison blind spot (P1)

1. **Blanket “exception → NaN”** (`verify.py:83-86`) contradicts D3 for
   outcomes that are `inf`:
   - `def f(x): return 1.0 / x` with `verify=(array([0.0, 2.0]),)` →
     vectorized `[inf, 0.5]`, expected `[nan, 0.5]` → **verification failure**.
   - Same for `math.log(0)` (`-inf`), `math.exp(1000)` (`inf`),
     `math.floor(inf)` (`inf`), and complex-producing `pow`.
2. **Float64 comparison hides integer errors** (`:91-100`):
   `verify_match(lambda x: x + 1, lambda x: x, (array([2**62], dtype=int64),))`
   **passes**; both sides round to the same float64.
3. **Frozen globals vs cache hits** (`api.py:180-195`): D8 freezes globals at
   first compilation; a cache hit still re-runs `verify` against the current
   original. `SCALE = 3.0`, verify; then `SCALE = 5.0`, verify again →
   `verification failed (got 3.0, expected 5.0)`, contradicting both
   documented behaviors.

Secondary: `_ATOL/_RTOL = 1e-9` applies to integer-like outputs; only
positional example args are supported (no kwargs, no scalar-only pinned
examples); failures report only aggregate `got`/`expected`.

**Proposals.** Model D3 explicitly (`ZeroDivisionError`/`OverflowError` →
documented `±inf`; domain errors/complex pow → `NaN`); compare integer/bool
results exactly before any cast; store the compile-time scalar results or
frozen environment in the canonical cache; support `verify=(args, kwargs)` and
pinned scalar examples; report the first mismatching index/value.

---

## F7 — Closure captures lock the generated function to one backend (P1)

D8 turns captured arrays into hidden keyword parameters with defaults.
`array_namespace` is then called with the caller's arrays **and** the captured
NumPy arrays, which fails across backends:

```python
ARR = np.asarray([1.0, 2.0])


def f(x):
    return x + ARR


vectorize(f)(array_api_strict.asarray([1.0, 2.0]))
# -> TypeError: Multiple namespaces for array inputs
```

The same happens for captured NumPy scalars (`np.int64(3)`, `np.float32(0.5)`)
because `compat._is_array` classifies them as arrays; `np.float64` is classified
as a scalar only because it subclasses `float` (`extract.py:104-106`).
`tests/backends/test_jax_torch_compilation.py:69-82` works around this by
passing `kwargs={"ARR": conv(ARR)}` — confirming the limitation — but no
user-facing doc mentions it.

Related API surface: hidden closure parameters are part of the real call
surface (users can override them) yet absent from `__signature__`, so they are
undiscoverable and typos surface as `TypeError`. NumPy-scalar captures are
inconsistent (float64 frozen as a constant, int32/int64 as hidden array params)
and are not range-checked.

**Proposals.** Detect the namespace from the user-facing arguments only, then
convert captures with `xp.asarray(captured)` inside the generated body (or
wrap captured values lazily); alternatively document “captured arrays must
match the call backend” and expose the hidden parameters in introspection.
Treat NumPy scalar types as scalars (with int64 validation). Add
cross-backend closure tests that do **not** pass the hidden kwarg.

---

## F8 — The int64 literal invariant is not enforced end-to-end (P1)

The validator rejects source constants outside int64
(`frontend/validate.py:210-213`), but literals created later escape it:

- `int(1e30)` and `math.trunc(1e30)` fold to a huge `ir.Literal`
  (`lower/expressions.py:205-221`); generated source contains
  `1000000000000000019884624838656`. Returning it works, but
  `x + int(1e30)` evaluates in float64 → `[1e30]` instead of the exact
  `10**30 + 1`.
- Closure scalars are accepted without a range check (`extract.py:104-105`);
  `K = 10**30; return x + K` emits the huge constant and loses precision.
- Defaults are checked only for AST shape (`extract.py:51-57`); a 31-digit
  literal default is accepted and emitted. `_params_of` runs before
  `validate`, so defaults are never visited by the validator.
- The int64 minimum cannot be written: `x - 9223372036854775808` is parsed as
  `UnaryOp(USub, Constant(2**63))`, and the constant is rejected before the
  sign is considered.

**Proposals.** Centralize construction in `ir.int_literal(value)` enforcing
`[-2**63, 2**63-1]`; use it in `_cast_call`, closure classification, defaults,
and `_const_int`. Normalize/allow `-2**63` by checking the signed value at the
`UnaryOp` level. For genuinely out-of-range constants, reject (D10) or lower
to a documented float64 value; do not emit a Python big int into array
arithmetic.

---

## F9 — Numeric translations are not exact or consistent for runtime dtypes (P1)

- **`math.floor`/`math.ceil`** lower to `xp.floor/ceil` after forcing float64
  (`lower/expressions.py:223-235`). `math.floor(9007199254740993)` →
  `9007199254740992.0`; `math.floor(x) + math.ceil(x)` on int64 `2**53+1` is
  off by 2. Python returns exact ints.
- **`abs`** on int64 min overflows to a negative value; on bool arrays it
  returns a bool array (which feeds F1).
- **`round`** on bool arrays returns `float16` under NumPy and raises under
  strict; Python returns `int`.
- **`int()`/`math.trunc`** use `astype(..., int64)`: for float/uint64 values
  outside int64 this wraps or is backend-defined (`int(1e30)` → int64 min with
  a runtime warning; `int(uint64 2**63+1)` → negative). The divergence table
  does not list this.
- **`x ** -1`** with an int-typed *parameter* emits `xp.pow(x, -1)` and raises
  `ValueError` on NumPy, although scalar Python returns `0.5`. The
  negative-literal cast only fires when the base kind is provably int. A
  negative array exponent has the same problem.
- **Shifts** overflow/wrap at the array width while Python is unbounded
  (`2**62 << 2` → `0` vs `18446744073709551616`); only `+`/`*` folding is
  documented.
- **Zero divisors are dtype-dependent**:
  - `x % 0`: int64 → `0`, float64 → `nan`
  - `x // 0`: int64 → `0`, float64 → `inf`
  - Both NumPy and `array-api-strict` behave this way. `docs/semantics.md`
    says `x % 0` → `NaN`, and the `constfold` comment says div/mod/floordiv
    by zero reach “runtime inf/NaN”; neither holds for integer arrays.
- **Transcendentals** are libm-dependent; `math.sin` and `xp.sin` may differ by
  ULPs. The docs say “exact semantics” without stating the float tolerance.

**Proposals.** Add runtime dtype dispatch for `floor`/`ceil` (integer dtype →
identity/exact cast; float → `xp.floor`), `abs` (bool → int64; int64 min →
documented float64 or exact promotion), and `round`. Make `pow` dispatch on
runtime dtype when the exponent can be negative/non-integral. Decide and
document the fixed-width overflow and zero-divisor policy per dtype. State the
transcendental ULP tolerance in `semantics.md`.

---

## F10 — Diagnostics do not match the documented format (P2)

Docs show `VectorizationError: <file>:3:5: 'while' loops are not supported`
(`docs/api.md:78-84`, `docs/quickstart.md:88-90`). Actual output:

```
cannot vectorize 'bad': 1 unsupported construct found
  line 2, col 4: while loops are not supported (data-dependent control flow)
          while x > 0:
          ^
```

No filename; the line number is relative to the dedented function snippet
(`Diagnostic` in `errors.py:16-30`, `validate.py:100-106`); lowering errors
have no caret and sometimes only `(line N)` (`lower/env.py:104-109`);
`Diagnostic.col_offset` is a UTF-8 byte offset, so carets misalign on
non-ASCII lines. The fallback warning keeps only the first line, losing count
and details (`api.py:200-210`).

**Proposals.** Add `filename` and an absolute line offset
(`tree.lineno - 1 + relative`) to `Diagnostic`; compute display columns with
`len(line.encode()[:col].decode(...))`; route lowering errors through the same
renderer; include the full message (or first diagnostic) in the fallback
warning; update `api.md`/`quickstart.md`. Add a golden diagnostic test with a
multi-line function and non-ASCII comments.

---

## F11 — API, cache, concurrency, and fallback issues (P2)

- **Thread safety** (`api.py:77-89`): `_ACTIVE_HELPERS` is a process-global set.
  Two threads compiling different functions that share a helper can produce a
  false “recursive helper calls are not supported” error; caches have
  check-then-act races. Use a lock and/or a per-compilation stack, or key the
  active set by `(thread_id, callee)`.
- **Pinned compilation is not memoized**: `vectorize(f, namespace=np)`
  recompiles on every call, while `vec.with_namespace(np)` is memoized
  (`api.py:96-112`).
- **Fallback precision** (`fallback.py:40-63`): `xp.asarray(out)` uses the
  backend default dtype. With torch, a float64 input becomes float32;
  subnormal/small values can underflow to zero. The module calls itself a
  “correctness escape hatch”, so this matters. Test comments acknowledge
  “values, not dtype, are the fallback contract”, but the docs do not.
- **Fallback limitations**: captured closure arrays are not broadcast correctly
  in the element loop; all-scalar calls are rejected even with a pinned
  namespace, contradicting the pinning promise; callable objects without
  `__name__` fail in `make_fallback`; `get_source(fallback)` returns a comment.
- **Zero-argument functions** are rejected (`extract.py:75-76`), although a
  pinned vectorization could define a shape-free constant result.
- **`__annotations__`** are not copied to the generated function.
- **Hidden signature params** are not discoverable (see F7).

---

## F12 — Optimizer and runtime performance opportunities (P2/P3)

- **CSE is narrow**: `_CSE_ELIGIBLE = (Call, Where)` (`optimize/cse.py:19`), so
  repeated `BinOp`/`Compare`/`Logical` and helper `FuncCall`s are never shared;
  CSE is disabled entirely for any program containing a loop (`:33-39`).
- **`protect_domains` runs last** (`pipeline.py:73-77`) with no subsequent
  fold/CSE/DCE. It duplicates the argument expression in every clamp; a
  two-level `sqrt(log(sqrt(x)))` body grows from 569 to 1,456 characters (and
  nests multiplicatively). Re-run the optimizer after protection, or fold the
  clamp conditions while inserting them.
- **Every division goes through the Python helper** when operand kinds are
  unknown (`expressions.py:366-391`), even for two float64 arrays; it also
  upcasts float32 to float64. Measured overhead on 1M floats: ~0.68 ms vs
  ~0.62 ms (~10%), worse for small arrays. Add a fast path for provably float
  operands.
- **Both runtime helpers are injected into every generated module
  unconditionally** (`lower/__init__.py:91-104`); `_namespace` is reserved even
  when unpinned. Only inject what the IR references.
- **Constant propagation is not applied to general expressions**: literal
  facts are stored (`Kinds.literal`) but `constfold._const_fold_expr` does not
  substitute `Ref`s, so `y = 4; z = y * 2` emits runtime scalar arithmetic and
  `n = 3; for i in range(n)` is rejected despite D6’s “bounds known at
  generation time”. `range(True)` (valid Python) is also rejected.
- **True-division helper** runs a 55-step bit loop; `_u64_sub_exact` and
  signed-u64 paths call `xp.all(...)` several times, forcing GPU
  synchronization. Consider a fast path when both operands fit int64 (values
  ≤ 2**53 give exactly rounded float64 division) and dtype-level decisions
  where possible.
- **`_has_loop`** in `cse.py:22-29` has a redundant inner recursion; simplify
  to `any(isinstance(s, Loop) ...)`.
- **`optimize/__init__.py`** shadows the `cse`/`dce`/`constfold` submodule
  attributes with functions (documented NOTE); rename the pass entry points or
  avoid re-exporting them.

**Highest-leverage enhancement:** dtype-specialized compilation
(monomorphization). A `dtypes=`/`assume_float` option lets the compiler emit
bare operators and constant-fold dtypes, while the default exact path uses
runtime dispatch. This fixes F1 performance and simplifies generated source
for the common float case.

---

## F13 — Documentation defects (P2)

- `docs/api.md:78-84`, `docs/quickstart.md:88-90`: diagnostic format is wrong
  (F10).
- `docs/api.md:14` vs `docs/quickstart.md:72-77`: the table implies
  `fallback=True` only warns, but it also enables fallback; `strict=False` and
  `fallback=True` are equivalent triggers.
- `docs/api.md:44-46`: “signature, defaults, and kwarg names are preserved” is
  false for hidden closure parameters, `array_namespace`/`hasattr` parameters,
  and `__annotations__`.
- `docs/api.md:50` / `architecture.md:70`: `get_source` returns a comment for
  fallback wrappers.
- `docs/architecture.md:24-25`: “resolves lambdas and nested pure functions” —
  nested `def`s are rejected (`validate.py:36`); only lambdas passed as the
  target are resolved.
- `docs/architecture.md:61-63`: helper memoization is keyed by
  `(callee, protect, namespace)`, not `(callee, protect)`.
- `README.md` “Exact semantics” and `docs/semantics.md` bool/uint64 claims are
  contradicted by F1.
- `docs/semantics.md`: add `int()`/`trunc` outside int64, `abs(int64 min)`,
  `floor/ceil` of large ints, negative-exponent integer `pow`, shift overflow,
  zero-divisor behavior per dtype, `math.exp` overflow → inf,
  `math.floor(inf)` → inf, `round(nan)` → nan, `copysign` NaN payload/sign,
  transcendental ULPs, uint64/int64 merges → float64, JAX `x_enable_x64`,
  closure-capture backend locking.
- `docs/supported-subset.md`: state the dtype caveats for
  `int/float/bool(x)`, `math.floor/ceil`, shifts, and `**`.
- `docs/development.md` / `docs/api.md`: the NumPy 1.x claim is untested
  (`make lowest` floors numpy at 2.0); add a `numpy<2` CI job or drop the
  claim. CuPy is advertised but untested.
- `docs/index.md` “46 golden source snapshots” is 45 names + 1 pinned variant;
  clarify.
- `docs/development.md`: `make check` does not run fuzz/backends/docs/smoke/
  lowest, although it is described as “the gate”; list what each CI job adds.
- Add a **“Backend notes”** page: JAX x64, `array-api-compat` 1.15 for
  unpinned `torch.compile`, closure-capture namespace behavior, fallback dtype
  policy, CuPy status.

---

## F14 — Testing gaps and development-practice observations (P2)

The suite is green (and the gate is genuinely runnable), but it is
structurally unable to catch F1–F9:

- **No call-time dtype matrix.** Bool/uint64 coverage is dominated by
  expressions the compiler can *prove* (`(x > 0) + …`, `min(x, True)`); there
  is no test of `def f(x, y): return x + y` with bool, uint64/int64, or mixed
  inputs. Add a property test over dtypes (`float16/32/64`,
  `int8/16/32/64`, `uint8/16/32/64`, `bool`) × operator × compile shape,
  comparing to the scalar oracle with **exact** equality for integer/bool
  outputs.
- **`np.allclose` everywhere** (99 occurrences in behavior/features) hides
  integer and dtype errors. Use `array_equal`/`assert_equal` for integer and
  bool results; reserve tolerance for transcendentals.
- **The faulty `_vec_arith` add/mul branch is covered** by the existing test,
  but only with a positive result; line coverage (96%) is not semantic
  coverage. Extend `test_uint64_negative_literal_arithmetic_exact` with
  result-negative `add`/`mul`, operand order, and the `2**63` edge.
- **Fuzzer domain is float64-only** and `except TypeError: edge-skip`
  (`fuzz.py:200-205`) can mask real backend errors. Add int/bool/uint64
  values, dtype choices, helper calls, closures, and a strict-backend run;
  narrow the TypeError skip to the documented dead-lane case.
- **Golden snapshots** omit arithmetic-helper `+ - *` paths, mixed-kind
  merges, and fallback. Add a runtime-dispatch function, a mixed-kind merge,
  and a scalar-flag function.
- **Inspectability** pins the colliding filename; add same-name/two-module and
  canonical-vs-protected tests.
- **Verify tests** cover only sqrt-NaN; add division-by-zero, `log(0)`,
  exp overflow, complex pow, big-int exactness, frozen-global cache hits, and
  pinned scalar examples.
- **Closure/backend tests** explicitly override the closure kwarg for
  jax/torch; add a test that calls the default closure with a strict/torch
  input so the limitation is visible or fixed.
- **Perf gate** measures only `v + 1.0`; add division (helper path), a
  branch/loop kernel, and small-array dispatch overhead.
- **Docs snippets** in README/index are not executed; add a test that
  regenerates or extracts and compares them.
- **Stale test comment**: `test_preamble_normalizes_scalar_params` claims a
  preamble that does not exist; rename and assert actual behavior.
- **`make check` scope** excludes `make fuzz`, `make backends`, `make docs`,
  `make smoke`, `make lowest`; document this gap or widen the target.
- **No NumPy 1.x / CuPy jobs** despite documentation claims; add at least a
  `numpy<2` job or remove the claims.

---

## Suggested fix order

1. **P0 contract closure** (F1, F2): fix F1.3, add runtime dtype dispatch and a
   `_vec_where` merge helper, and adopt an explicit decorator policy. Land the
   dtype × operator property test first.
2. **P1 backend neutrality** (F3, F4, F5): parameter normalization or
   systematic wrapping, `not` coercion, name-resolution hygiene, unique
   filenames, closure-to-namespace conversion (F7), int64 literal validation
   (F8).
3. **P1 `verify` and numeric translations** (F6, F9): divergence modeling,
   exact integer comparison, frozen-cache semantics; runtime dispatch or
   documented limits for `floor/ceil/abs/round/int/trunc/pow` and zero
   divisors.
4. **P2 hygiene and performance** (F10–F12): diagnostics with file/absolute
   positions, cache/thread policy, fallback dtype policy, CSE scope,
   protect-then-optimize ordering, unconditional helper injection, constant
   propagation.
5. **Docs and guardrails** (F13–F14): rewrite the exactness contract as a
   dtype table, add backend notes, and land the dtype matrix, fuzz expansions,
   strict-backend scalar tests, and executable doc snippets.

The highest-value single change is the runtime dtype contract: it converts most
of F1, F7, F8, and part of F9 from “silent backend-dependent behavior” into
either exact results or a clear rejection, which is exactly the project's
stated D10 principle.

---

## Artifact index

- `artifacts/repro_dtype_contract.py` — bool/uint64/mixed-arithmetic,
  merge/phi, `_vec_arith` negative add/mul, casts, `floor`, `pow`, `abs`.
- `artifacts/repro_backend_neutrality.py` — strict-backend failures for loop
  indices, scalar conditions, `not`, bool arithmetic, closure captures, and
  integer zero divisors.
- `artifacts/repro_emit_verify.py` — linecache collision, decorator
  miscompile, `verify=` divergence/blind-spot/frozen-global cases, `range`
  shadowing, reserved-parameter keyword calls, `protect_domains` growth.
