# Reconciled repository review: array-vectorize

Date: 2026-10-23. Compiler revision reviewed:
`0e4bb6b64ae72ae369bbd61c921cf8673ec8d1a4`.

This report combines two independent reviews, subsequent cross-review, corrected
scalar oracles, and additional reproductions. It is the reconciled assessment;
the original reports remain historical evidence. The final approval receipt is
maintained separately in
[CONSENSUS.md](review-artifacts/reconciliation/CONSENSUS.md).

- Reviewer A: Paseo `3ae67309-cf05-49d7-961e-c13ec5ea721e`, original report
  [PROJECT_REVIEW.md](PROJECT_REVIEW.md), committed with artifacts in `33efb77`.
- Reviewer B: Paseo `3dd792b2-ca06-4084-911e-c74113895e1e`, original report at
  `a531fe0`, preserved as
  [architecture and exactness review](review-artifacts/reconciliation/peer-review/2026-10-23-architecture-and-exactness-review.md).
- The evidence archive and its interpretation notes are indexed in
  [reconciliation artifacts](review-artifacts/reconciliation/README.md).

## Assessment

**CHANGES REQUESTED on correctness.** The compiler has a coherent architecture,
strong development checks, and carefully implemented runtime algorithms. However,
accepted functions still produce wrong values on ordinary bool and mixed-integer
inputs. Other failures concern comparisons, numeric boundaries, decorators,
domain protection, name resolution, backend argument normalization, API metadata,
and verification tools.

The central architectural problem is that unknown runtime kinds sometimes take
raw backend operator paths. A parameter being statically unknown does not prove
that Python semantics and backend promotion agree. Routing more operations through
the existing helper alone is insufficient: the helper's negative-result add/mul
paths also need correction.

The priorities are to strengthen the scalar oracles, repair demonstrated semantic
failures, retain a frozen compilation artifact for variants, and align public
claims with supported backend/dtype combinations. Broader syntax and optimization
work should follow those steps.

No compiler implementation or existing tests were changed during reconciliation.
This review proposes fixes; it does not claim they have been implemented.

## Project understanding and strengths to retain

`vectorize(f)` eagerly extracts source, validates a restricted scalar subset,
lowers it into an explicit IR, optimizes, emits Python AST, executes the generated
module, and attaches inspectability metadata. Array-dependent branches become
selects; constant-trip loops retain Python loop structure with carried bindings.
Call-time runtime helpers supply semantics that bare Array API operations cannot
provide safely.

The principal boundaries are sound:

- `api.py` owns options, caches and fallback; `pipeline.py` orchestrates strict
  compilation through an injected helper-vectorizer callback.
- `frontend/` separates source/capture extraction, lambda identification and
  structural validation. Lowering supplies further semantic checks.
- The IR is dependency-free, while the runtime is a stdlib-only leaf. Layering
  tests make these architectural boundaries executable.
- `lower/` separates expressions, statements, names, loops, kind facts and helper
  calls, though its shared state and repeated loop-body lowering require care.
- AST emission avoids source-template escaping errors. Source registration,
  signatures, docstrings and original-function markers are useful public features.
- Integer true division uses an explicit rounding algorithm with compiled-backend
  regressions. Result-aware subtraction and exact min/max integer selection are
  useful foundations, even though they do not cover every call path.
- Unit, behavior, strict-backend, differential, golden, performance and infrastructure
  tests complement lint, four type checkers, dependency-floor tests, docs checks,
  platform CI, fuzzing and isolated installed-artifact release smoke tests.

These strengths favor incremental, regression-backed changes over a rewrite.

## Evidence and limits

| Validation | Observed result |
|---|---|
| Independent default-environment `make check` runs | 1,064 passed, 97 skipped; combined statement/branch coverage 96%; lint and all four type checkers passed |
| Reviewer A's later `make check` with optional backends installed | 1,160 passed, 1 skipped; coverage 96% |
| `make docs`, `make smoke` | Passed in both reviews |
| `make fuzz` | Five seeds, 2,000 generated cases: 1,482 checked, 2 rejected, 516 edge-skipped |
| `make backends` | 96 passed, including JAX/PyTorch compilation tests |
| `make lowest` | 1,064 passed, 97 skipped at direct default dependency floors |
| Build and installed-artifact smoke | Wheel and sdist built with `uv build --no-sources` and separately smoke-tested outside the checkout |
| Numeric boundary matrix | 17,312 evaluated backend lane checks; 1,148 excluded scalar exceptions or results outside its selected integer-result envelope |
| Enabled compile benchmark | Approximately 506 ns mean, revealing cache lookup rather than cold compilation (R21) |
| Reconciliation | Peer scripts rerun, disputed expectations checked against scalar Python, and additional cases reproduced on NumPy and strict |

The main environment was Python 3.12.13, NumPy 2.5.3, `array-api-compat` 1.15.0,
`array-api-strict` 2.6.1, JAX 0.11.2 and PyTorch 2.14.1+cpu. Optional backend
installation explains the different default-suite skip counts. The isolated
packaging smoke resolved NumPy 2.5.4.

The boundary matrix exercised bool/int64/uint64/float64, Python scalar operands,
signed zeros, subnormals, infinities, NaNs, int64 bounds, `2**53` boundaries, and
uint64 values above `2**63`. Other probes exercised control flow, helper calls,
captures, CSE, API options, rejection paths and controlled concurrency. Raw matrix
mismatch totals include repeated symptoms and documented divergences and are not
counts of independent defects.

Integer true division, signed-zero-aware CSE, basic protection contexts, and
several signed/unsigned min/max selections held in the exercised cases. This is
useful counterevidence, not a proof of all combinations.

Neither review locally exercised Windows, Python 3.13/3.14, CuPy/GPU behavior,
NumPy 1.x, or optional backend dependency floors. CI configurations were inspected.
This is an extensive counterexample-based review, not formal verification.

## Severity and reproduction conventions

- **P1:** silent wrong values or a central semantic guarantee violated.
- **P2:** accepted combinations, backend/API behavior, or validation tools fail.
- **P3:** narrower edge, diagnostic, benchmark, or usability issues.

Documented limitations and proposals are separated from these defects. In
particular, branch promotion, mixed int/float min/max promotion, and stated
fixed-width overflow behavior are not relabeled as undisclosed miscompiles.

Save reproduction functions in real `.py` files and run with `uv run python`.
Unless imports are supplied, the examples below use:

```python
import math
import numpy as np
import array_api_strict as strict
from array_vectorize import vectorize

xp = np  # repeat with strict
```

## Agreed implementation and API defects

### R01 — P1: unknown bool and unsigned unary operands use backend semantics

**Code:** `lower/__init__.py:102`, `lower/expressions.py:71,330,366`.

```python
def add(x, y):
    return x + y

def complement(x):
    return ~(x > 0)

print(vectorize(add)(xp.asarray([True]), xp.asarray([True])))
print(vectorize(complement)(xp.asarray([-1, 1])))
```

Expected Python integers: `[2]` and `[-1, -2]`. NumPy returns bool `[True]` and
`[True, False]`; strict rejects addition and returns the same wrong complement.
Direct bool parameters also break unary `+`/`-`, and `abs(bool)+abs(bool)` can
produce a saturated boolean rather than integer 2.

Unsigned unary operations have the same missing semantic dispatch. At uint64
value 1, `-x` returns `18446744073709551615` instead of `-1`, and `~x` returns
`18446744073709551614` instead of `-2`, on both backends. The expected negatives
fit int64; this is not the documented same-int64 overflow case.

**Proposal:** conservatively track possible runtime kinds for unknown parameters
and dispatch Python numeric operations accordingly. Normalize boolean/unsigned
unary operations before backend execution. Test parameter, alias, comparison,
helper-result and loop-carried forms. Python deprecates `~bool`; explicitly
rejecting that form is preferable to accepting it with different semantics.

### R02 — P1: ordinary mixed arithmetic bypasses the exactness lattice

**Code:** `lower/expressions.py:366–407`.

```python
def subject(x, y):
    return x - y

x = xp.asarray([2**63 + 1], dtype=xp.uint64)
y = xp.asarray([2**63 - 1], dtype=xp.int64)
print(vectorize(subject)(x, y))
```

Expected `[2]`; NumPy produces float `[0.]`, while strict raises a promotion
`TypeError`. The result fits an integer dtype, so the unrepresentable-result
approximation does not apply. Other examples include `(2**64-1) % 3` with
uint64/int64 operands producing NumPy `1.0` rather than 0. Mixed integer bitwise
operations can raise even when the Python result fits exactly.

**Proposal:** route unknown/mixed risk operations through correct runtime dispatch
or guard/reject unsupported combinations. Keep proven-safe fast paths. Regression
tests must exercise the public compiler, not only the runtime helper.

### R03 — P1: the runtime helper itself wraps representable negative add/mul results

**Code:** `runtime/arith.py:others_intish` add/mul branch, around lines 275–277.

```python
def add_negative(x):
    a = min(x, True)
    return a + -1

def multiply_negative(x):
    a = min(x, True)
    return a * -1

print(vectorize(add_negative)(xp.asarray([0], dtype=xp.uint64)))
print(vectorize(multiply_negative)(xp.asarray([1], dtype=xp.uint64)))
```

Both backends return uint64 `18446744073709551615`; both scalar results are `-1`.
These forms reach `_vec_arith`, so merely broadening its use will not fix them.
Existing tests cover the branch with positive results and miss the negative partition.

**Proposal:** make negative-scalar addition and multiplication result-aware as
subtraction already is. Guard magnitudes before products and preserve int64-min
without signed-overflow assumptions. Apply any documented float fallback to the
original operands, never an already wrapped product. Test both operand orders,
negative/zero/positive results and magnitude boundaries.

### R04 — P1: comparisons lose Python's exact mixed-number behavior

**Code:** `lower/expressions.py:109`, Compare emission in `codegen/emitter.py`.

```python
def subject(x):
    return x == 9007199254740993

print(subject(float(2**53)))
print(vectorize(subject)(xp.asarray([float(2**53)], dtype=xp.float64)))
```

Expected `False`; both backends return `[True]`. The literal is rounded against
the float array. Mixed int64/float64 array equality/order also miscompares on
NumPy, while strict rejects several mixed dtype combinations. Public `verify=`
does catch this specific boolean mismatch if those example inputs are supplied.

**Proposal:** implement exact mixed comparisons or reject unhandled combinations;
casting both sides to float64 is insufficient. Cover all six relations, literal
order, signed/unsigned pairs, infinities, NaNs and precision boundaries.

### R05 — P1: generic decorator unwrapping silently discards behavior

**Code:** `frontend/extract.py:43`, `resolve_original`.

```python
import functools

def decorate(fn):
    @functools.wraps(fn)
    def wrapper(x):
        return fn(x) + 100.0
    return wrapper

@decorate
def subject(x):
    return x * 2.0

print(subject(3.0))
print(vectorize(subject)(xp.asarray([3.0])))
```

Scalar output is 106; both vectorized outputs are 6. `inspect.unwrap` follows
arbitrary `__wrapped__` links before compilation. Source and metadata can then
refer to different objects.

**Proposal:** distinguish this library's own vectorization markers from arbitrary
decorators. Adopt an explicit policy: compile supported wrapper behavior, reject
unsupported wrappers, or require an opt-in to compile the inner callable. A
behavior-changing wrapper must not be silently discarded.

### R06 — P1: protected-domain liveness misses nested loop reads

**Code:** `optimize/protect_domains.py:167–169`.

```python
def subject(x):
    a = math.sqrt(x)
    s = 0.0
    for i in range(1):
        for j in range(1):
            s = s + a
    return a if x > 0 else s

x = xp.asarray([-1.0, 1.0])
with np.errstate(invalid="ignore"):
    print(vectorize(subject)(x))
    print(vectorize(subject, protect_domains=True)(x))
```

Both backends change `[NaN, 1.]` into `[0., 1.]`. Scalar Python raises on the
negative lane, but the documented vectorized domain-error result is NaN;
protection promises not to change that result.

The pass scans only immediate `Binding` reads inside a loop and omits nested
`Loop` bodies. Its bound-name traversal is recursive, but its read traversal is not.

**Proposal:** recursively collect all nested statement and bound reads before
propagating liveness. Check protected/unprotected equivalence with NaN and zero-sign
assertions, not only approximate finite values.

### R07 — P1: binding resolution is skipped for `range` and some builtin calls

**Code:** `frontend/validate.py:validate_for`, `lower/loops.py:96`,
`lower/expressions.py:_lower_call`.

```python
def range(n):
    return (0, 10)

def subject(x):
    for i in range(2):
        x += i
    return x

print([subject(x) for x in [1, 2]])
print(vectorize(subject)(xp.asarray([1, 2])))
```

Expected `[11, 12]`; both backends return `[2, 3]`. The custom tuple-returning
helper is unsupported, so rejection is the correct fallback if it cannot be lowered.

Related closure/global scalar shadowing such as `abs = 5` or `int = 5` can compile
as the builtin even though scalar execution raises `TypeError`. Parameter/local
`range` shadowing also needs a positioned rejection rather than malformed generated
behavior. These associated rejection issues are P2 in isolation.

**Proposal:** require binding identity before special handling; check locals,
parameters, closures and globals before builtin dispatch. Test all builtin-looking
names, not just `range`.

### R08 — P1: integer casts and integer-valued math corrupt representable boundaries

**Code:** `lower/expressions.py:205–233`, `frontend/tables.py:49,50,69,89`;
raw unary/abs emission.

| Scalar operation and input | Scalar expectation | NumPy and strict behavior |
|---|---|---|
| `int(uint64(2**63+1))`, `math.trunc(...)` | `9223372036854775809` | `-9223372036854775807` |
| `int(float(2**63))` | `9223372036854775808` | int64 minimum |
| `math.floor(int64(2**53+1))`, `math.ceil(...)` | `9007199254740993` | floating `9007199254740992` |
| `abs(int64(-2**63))` | positive `9223372036854775808` | negative int64 minimum |

The unsigned or signed integer dtype needed for these chosen expectations exists.
Unchecked int64 casts, forcing integer math through float64, and fixed-width unary
operations lose the value before subsequent code can recover it.

**Proposal:** preserve integer identity operations; use checked casts; dispatch
integer-valued math without early float conversion; handle magnitude promotion
at the signed minimum. Guard or reject values outside every supported representation
under an explicit policy. Do not equate a documented output dtype mapping with
permission to silently wrap its values.

### R09 — P2: individually supported numeric constructs fail when composed

**Code:** `frontend/tables.py`, `lower/expressions.py:223,403`.

```python
def floor_bit(x):
    return math.floor(x) & 1

def reciprocal(x):
    return x ** -1

print(vectorize(floor_bit)(xp.asarray([1.5])))
print(vectorize(reciprocal)(xp.asarray([2], dtype=xp.int64)))
```

Expected `[1]` and `[0.5]`; both backends raise `TypeError` and `ValueError`
respectively. `ceil`/`round` integer composition has the same issue. The negative
power mitigation casts only bases already proven integer and misses parameters
and some bound exponents.

**Proposal:** track Python result kind independently of backend representation;
materialize integer/floating representations at use sites or reject combinations
that cannot be exact. Add composition properties, not only standalone function checks.

### R10 — P2: strict backends receive invalid raw scalar or numeric logical arguments

**Code:** `lower/sanitize.py`, `lower/expressions.py`, `codegen/emitter.py`.

Independently reproduced strict failures include:

- `if flag > 0: return x; return -x` with an array `x` and Python scalar flag:
  the raw bool condition reaches `xp.where`.
- A constant-trip loop accumulating `i//2 + i%2 + i**2`: raw Python loop indices
  reach `xp.floor_divide`, `xp.remainder` and `xp.pow`. Scalar output is 7 for
  `range(3)`; strict raises.
- `return not x` on float arrays: `xp.logical_not` requires boolean dtype in
  strict. Scalar expectation for `[0., 2.]` is `[True, False]`.
- `return math.sqrt(-1.0) + x`: failed constant folding leaves a raw float math
  argument; NumPy returns documented NaN, strict raises `TypeError`.

**Proposal:** systematically normalize Array API call arguments, conditions and
numeric truthiness. For numeric `not`, use an exact truthiness coercion before
boolean operations. Ensure declined constant folds remain runtime-valid. Remove
stale comments/tests claiming a parameter-normalization preamble already exists.

### R11 — P2: scalar-only helper arguments lose the caller's namespace

**Code:** `lower/helpers.py:35,43`.

```python
def helper(y):
    return y * y

def subject(x):
    return helper(2) + x

print(vectorize(subject)(xp.asarray([1, 2])))
```

Expected `[5, 6]`; both backends raise because the compiled helper detects a
namespace only from raw scalar 2. The caller already has the required namespace.

**Proposal:** pass runtime namespace context to helpers, or inline/fold them when
sound. Test literal-only, defaulted and zero-dimensional helper arguments.

### R12 — P3: `min`/`max` signed-zero ties select a different operand

**Code:** `runtime/minmax.py:40–47`.

```python
def subject(x, y):
    return math.copysign(1.0, min(x, y))

print(vectorize(subject)(xp.asarray([0.0, -0.0]),
                        xp.asarray([-0.0, 0.0])))
```

Expected `[1., -1.]`; both backends return `[-1., 1.]`. `max` behaves similarly.
This is an observable finite tie difference; the NaN and mixed-kind promotion
divergences do not cover it.

**Proposal:** retain the first equal operand in the comparison/select fold,
separately preserving the intended NaN policy. Test zero signs and argument order.

### R13 — P2: integer zero remainder contradicts the documented NaN result

**Code:** floor/mod lowering and runtime dispatch; `docs/semantics.md` divergence table.

`def subject(x): return x % 0` returns integer `[0]` for int64 `[4]` on both
backends. The documented result for `x % 0` is NaN; float arrays do produce NaN.
Integer `x // 0` also yields 0, while floating division yields IEEE values.

**Proposal:** enforce the stated `% 0` policy across dtypes or change it explicitly.
Specify floor-division zero behavior per dtype as well; its table entry is less
explicit, so the `// 0` observation is a policy gap rather than a separately proven
violation of an unambiguous table row. Rejection is preferable to accidental zeros.

### R14 — P2: namespace variants re-extract and recapture frozen environments

**Code:** `api.py:103–111`.

```python
SCALE = 2

def subject(x):
    return x * SCALE

v = vectorize(subject)
SCALE = 3
print(v(np.asarray([1])))
print(v.with_namespace(np)(np.asarray([1])))
```

Observed `[2]` and `[3]`. `docs/api.md` says variants do not rerun verification
because "the body code is unchanged, only the `xp` binding differs." Implementation
recompiles the original, recapturing globals/nonlocals and potentially helper graphs.

**Proposal:** derive variants from one immutable compilation artifact containing
resolved source, effective signature, captures, IR and helper artifacts. Cache
variants by artifact identity. Expose recapturing recompilation separately if wanted.

### R15 — P2: reserved public parameter spellings no longer bind as advertised

**Code:** `ir/ssa.py:28,62`, signature emission, `emit.py:78`.

For `def subject(array_namespace): return array_namespace + 1`, introspection
advertises `(array_namespace)`, but `v(array_namespace=np.asarray([2]))` raises
an unexpected-keyword `TypeError`. Positional calls work. `hasattr` has the same
reserved-name treatment; `__signature__` does not alter the actual Python binder.

**Proposal:** retain public argument names and allocate collision-free internal
aliases, or reject the unsupported signature explicitly. Verify keyword and
keyword-only calls through all internal-name collisions.

### R16 — P3: effective defaults disagree with generated source defaults

**Code:** `frontend/extract.py:59`, signature emission, `emit.py:78`.

```python
def subject(x, y=2):
    return x + y

subject.__defaults__ = (7,)
v = vectorize(subject)
print(subject(1), v(np.asarray([1])))
```

Observed scalar 8 and vector `[3]`, while introspection advertises `y=7`.
Mutation precedes first compilation, so first-compilation freezing does not explain
the mismatch. AST defaults and the callable's signature are separate sources of truth.

**Proposal:** freeze validated defaults from the effective callable signature and
use the same data for code generation and metadata, or reject disagreement. Include
`__kwdefaults__` tests.

### R17 — P2: namesake compilations overwrite source inspection and tracebacks

**Code:** `emit.py:46,66`.

Two factories or modules each defining `subject`, one returning `x+1`, another
`x*9`, receive the same `<array_vectorize:subject>` filename. Compiling the second
makes `inspect.getsource(first_vec)` show the second's source while `.source`
remains correct. Option variants and lambdas also collide. Traceback text can point
at the wrong generated statement.

**Proposal:** use a unique/content-addressed generated filename and test multiple
simultaneously live namesakes and variants. Replace the test that pins the exact
colliding filename with an inspectability/uniqueness assertion.

### R18 — P2: fallback loses Torch precision; related option/callable cases are incomplete

**Code:** `fallback.py:42,63` and metadata construction.

For a scalar identity expressed with an unsupported `while`, `fallback=True`
returns Torch float32 from a float64 input. At `1.0000000001`, expected output is
`1.0000000001`; actual output is `1.0`. This changes values, not only dtype.
The demonstrated precision defect is Torch-specific; it is not generalized to
NumPy or JAX.

Pinned fallback still rejects all-scalar calls, despite the broad pinning promise.
A callable object with `__call__` but no `__name__` also triggers `AttributeError`
during fallback construction. That narrower callable-label failure is P3.

**Proposal:** define precision/range/device and empty-output policies for fallback,
construct output accordingly, and test values off the float32 grid. Support or
clearly reject pinned scalar/callable cases; use safe metadata labels rather than
assuming every callable is a plain function.

### R19 — P2: verification erases exact distinctions and crashes on complex-producing pow

**Code:** `verify.py:81–102`.

The checker casts both sides to float64 and applies one tolerance. On both backends,
these intentionally wrong-result mutations pass:

```python
from array_vectorize.verify import verify_match

def identity(x):
    return x

verify_match(lambda x: x + 1, identity,
             (xp.asarray([2**53], dtype=xp.int64),))
verify_match(lambda x: -x, identity,
             (xp.asarray([0.0], dtype=xp.float64),))
```

These isolate checker blind spots; they are not additional compiler miscompile
claims. Exact integer distinctions, kind distinctions and zero signs are erased.

Separately, `def subject(x): return x**0.5` with `verify=(array([-8.0]),)`
raises an uncaught `TypeError` on both backends when the scalar complex result
reaches float construction. The vectorized operation's documented result is NaN.

**Proposal:** compare bool/integer values without lossy casting, check zero signs
where required, apply only justified floating tolerances, and explicitly model
documented complex-producing cases. Mutation-test value/kind/sign/NaN/shape/error
classification. Documented NaN-only exception handling and frozen-cache verification
policy are discussed separately below.

### R20 — P2: the fuzzer can pass wrong values and suppress unconditional errors

**Code:** `fuzz.py:175,196,201,226`.

Replacing the compiled result of scalar `return 2.0` with all-`True` bool output
makes `_check_case` return `"ok"`: it converts both results to bool whenever either
is boolean. Replacing the compiled callable with an unconditional `TypeError`
makes it return `"edge-skip"`, without proving a documented dead-lane cause.

The CLI also returns success after zero cases for `--cases 1 --seconds 0`, despite
documenting case count as overriding time. `--cases 0` becomes unlimited through
`args.cases or inf`. These CLI issues are P3.

**Proposal:** compare kinds/values accurately, classify permitted divergences
narrowly, fail unexpected exceptions, and add bool/integer/uint64 and strict-backend
strategies. Validate bounds and termination semantics. Retain minimized source,
inputs, backend versions and seed as failure artifacts. Track skip growth.

### R21 — P3: the compile benchmark times canonical-cache hits

**Code:** `tests/perf/test_bench.py:99`, `api.py` canonical-cache path.

`benchmark(vectorize, subject)` repeatedly uses one function object. The first
call compiles; later calls return the cached function. The enabled run measured
approximately 506 ns mean across 707 rounds, not compilation of a branch/loop body.

**Proposal:** separate cold compilation, warm helper compilation, hot cache lookup,
first JIT compilation and runtime kernels. Arrange fresh function/artifact identity
outside the timed interval and verify compilation-stage invocation counts.

### R22 — P3: diagnostics lack promised file/absolute positions and consistent collection

**Code:** `frontend/info.py`, `frontend/validate.py`, `lower/env.py:106`, `errors.py`.

An unsupported loop inside a function defined at file line 40 reports snippet-relative
`line 2, col 4`, without a filename. Extraction failures can have no diagnostics;
lowering failures often have only a relative line. AST byte offsets also need
conversion before being used as display columns for Unicode text.

The docs promise file/line/column/caret locations and collected violations more
broadly than implementation provides. Negative defaults such as `y=-1` are also
rejected as non-literals, a narrower usability/documentation issue.

**Proposal:** retain original source offsets, unify structured diagnostics across
stages, define collection boundaries, and test decorated/nested/Unicode source.
Support signed scalar defaults or explicitly document their exclusion.

### R23 — P2: independent concurrent compilations can be diagnosed as recursion

**Code:** `api.py:72–89`, global `_ACTIVE_HELPERS`.

A controlled scheduling probe pauses one helper compilation after it enters the
active set. Compiling a different outer function using the same helper then raises
`VectorizationError: recursive helper calls are not supported`; the first compilation
completes normally when released. The probe changes scheduling through a temporary
runtime hook, not compiler source or recursion semantics.

**Proposal:** use per-compilation/context-local recursion tracking and synchronize
cache publication as needed. Define the concurrency contract explicitly. This is a
reproduced false diagnostic, not merely an inferred cache race.

### R24 — P2: the signed int64 minimum literal is rejected before its sign is applied

**Code:** `frontend/validate.py` Constant validation; literal construction in lowering.

`def subject(x): return -9223372036854775808` is rejected because the AST contains
`UnaryOp(USub, Constant(2**63))`. The signed result is within the claimed int64
range. In contrast, `x - 9223372036854775808` contains a positive out-of-range
constant and is not a valid demonstration of this specific defect.

**Proposal:** validate signed literal values at the appropriate AST boundary and
centralize literal validation across stages. Huge defaults/captures/computed constants
also expose policy inconsistencies; those are qualified below rather than treated
as proof that all big-integer results must be exact.

## Contract limitations and coherence proposals

These concerns are valid and worth addressing, but are not additional P1
miscompile counts. The existing divergence rules materially affect interpretation.

| Concern | Observed behavior and qualification | Proposal |
|---|---|---|
| Mixed-kind branch merges | `xp.where` promotes int64/uint64 branches to float64. A selected huge unsigned value can lose precision even when the other branch is positive and all selected values would fit uint64. Branch promotion is explicitly documented. A huge unsigned/negative pair can fit no single integer dtype at all. | State the precision consequences explicitly; consider exact result-aware merges where representable, with a `_vec_where`-style helper. Define fallback/rejection for impossible common representations. |
| Mixed int/float min/max | Large integer winners may lose precision after float64 promotion. The mixed-sibling divergence already covers that promotion; it is separate from the finite signed-zero tie defect R12. | Explain the loss and consider exact selection/comparison improvements where feasible. |
| Fixed-width overflow | Same-int64 `max+1` and related fixed-width cases need careful classification against the documented lattice. Do not claim all such outputs violate an unbounded-integer guarantee. Unary/result-representable cases in R01/R08 are stronger counterexamples. | Publish operation/dtype/range tables covering unary, bitwise, shifts, casts, division and arithmetic; reject unclear cases until a deliberate policy exists. |
| Verifier exception policy | `verify=` explicitly expects NaN for scalar exceptions. Therefore valid vectorized `1/0 -> inf` can fail verification; this is documented, although awkward beside D3. | Align exception handling with operation-specific divergences, or clearly delimit NaN-only verification. Do not map every overflow/division exception to infinity without operation context. |
| Verifier and frozen captures | Cache-hit verification runs against the current scalar callable, while compiled captures remain frozen. Changing a global can produce a mismatch without a compiler regression; both behaviors are documented. | Define whether verification checks the frozen program or current callable. Store a frozen oracle/artifact if that is the chosen meaning; report stale captures clearly otherwise. |
| Captured-array portability | A NumPy capture included in namespace detection prevents a strict/Torch/JAX call unless replaced. Hidden capture overrides in backend tests expose this limitation. | Document backend/device/aliasing constraints and hidden overrides; offer deliberate opt-in conversion rather than silently copying captures. |
| NumPy scalar capture classification | `np.int64`/some float types can become hidden array captures while `np.float64`, a Python-float subclass, is frozen as a scalar. | Normalize supported scalar captures consistently with range validation, or document the distinction. |
| Oversized generated constants | `int(1e30)`, huge defaults and global integers can bypass the source Constant range check; array operations may then raise backend errors. Float64 `x + int(1e30)` correctly matches scalar float arithmetic. Some integer expectations exceed every available dtype and fall under approximation policy. | Enforce a coherent range policy at literal construction/default/capture boundaries; classify runtime rejection versus documented approximation. Do not use the incorrect float oracle as numeric-defect evidence. |
| Float32 true division | Parameter float32 division promotes to float64 through the helper, while several other operators retain float32. | State whether division intentionally promotes and test the choice across backends; evaluate performance/device consequences. |
| Signature/source surface | Hidden capture parameters are absent from the advertised scalar signature; annotations are not copied as a metadata dictionary; fallback `.source` is a comment. | Distinguish public signature from executable hidden bindings; document metadata/source limitations or expose them through a dedicated inspection API. |
| Constant-trip bounds | Locally assigned constant names are not generally propagated into loop bounds; bool bounds and zero-argument pinned constants are also rejected. | Expand exact constant propagation where worthwhile or tighten supported-subset wording. These are enhancement/rejection-policy issues, not silent miscompiles. |

## Architecture, performance and development proposals

The following proposals are jointly assessed as worthwhile options, not already
implemented fixes or proven performance wins.

| Proposal | Purpose and safeguards | Priority / approximate size |
|---|---|---|
| Immutable compilation artifact | Freeze source, effective signature, captures, helper graph, IR and options once. Derive variants without recapture and give caches an unambiguous identity. | High / medium-large |
| Explicit abstract kind lattice | Represent possible runtime kinds, Python result kind, raw-scalar status and backend dtype separately. Unknown must not mean safe native operation. | High / large |
| Semantic operation registry | Add result kind, argument normalization, domain and exceptional behavior to current name/arity metadata; keep lower/fold/runtime/verify decisions consistent. | High / medium |
| IR invariant validator | Check definition/use order, name uniqueness, loop scope, argument normalization and kind invariants after passes in tests/debug mode. | Medium / medium |
| Optional dtype specialization | Let a guarded dtype contract produce simpler fast code while default execution remains exact or rejects. Avoid an unchecked assumption mode silently weakening guarantees. | Medium / large |
| Cache lifecycle and observability | Strong unbounded canonical/helper/pin caches retain discarded functions, captures and namespaces; a weak-reference probe confirms retention. Unique filenames also increase source-cache entries. Define clearing/session scope, sizes and cleanup; weak keys alone are insufficient when values refer back to keys. | Medium / medium |
| Exportable source bundle | Generated text can depend on injected runtime/user helpers and hidden captures. Export those bindings/dependencies or qualify paste-and-run claims. | Medium / medium |
| Loop-aware optimization regions | CSE currently excludes programs containing loops and only shares selected node kinds. Start with safe straight-line regions and measured duplicate costs; preserve dominance, carried state and failure semantics. | Low / medium-large |
| Protection code-growth control | One nested-math example grew from 569 to 1,456 source characters. That observation is performance/code-size evidence, not a contractual 2x limit. Consider safe post-protection simplification/CSE with equivalence checks. | Medium / medium |
| Helper injection and import hygiene | Inject only helpers used by the IR; remove avoidable package function/submodule name collisions and small redundant traversals as maintenance permits. | Low / small-medium |
| Backend cost model | Runtime `bool(xp.all(...))` decisions can synchronize devices; integer division has a fixed bit loop and first-JIT costs. Benchmark representative eager/compiled/device cases before choosing range/dtype fast paths. | Medium / medium-large |

Do not substitute these structural proposals for fixing the concrete repros. In
particular, broad runtime dispatch requires R03 to be corrected at the same time.

## Testing and development practices

### Build discriminating semantic oracles

The green gate and 96% coverage coexist with short wrong-result programs. The gap
is distribution of semantic coverage, not simply the number of tests or checkers.

The shared Hypothesis helper in `tests/differential/test_scalar_equivalence.py`
often passes NumPy scalar elements to the scalar function, which can inherit
backend overflow/promotion instead of Python scalar behavior. Approximate comparisons
also hide integer, kind and signed-zero defects.

Create a contract-aware lane oracle that converts backend elements to plain Python
scalars, compares bool/integer outputs exactly, checks NaN and zero sign, and applies
narrow named divergences. Share applicable policy across tests, fuzzing and public
verification, but independently mutation-test the shared checker so one bug cannot
invalidate every evidence layer.

Generate a matrix over operation × input kind × operand source × backend × control
flow. Sources should include direct parameters, literals, captures, aliases, helper
results, selects and loop-carried values. Add representation-boundary and negative-
result partitions; R03 shows why exercising a line with positive results is insufficient.

### Preserve independent backend and option checks

Most JAX/PyTorch compilation tests compare compiled generated code with the same
generated function running eagerly. Retain those tracing tests and add scalar-Python
checks for representative boundaries, because a shared compiler mistake can pass both.

Use pairwise coverage across namespace, protection, fallback, verification, cache state
and capture mutation. Include scalar flags/helpers, zero-dimensional and empty arrays,
broadcasting, keyword-only calls, device behavior and concurrently live variants.

The test named `test_verify_detects_mismatch` currently demonstrates successful 2-D
verification; rename it and add an actual failure assertion. Add deliberate wrong
value/sign/kind/shape/exception mutations to verifier and fuzzer tests.

### Improve evidence from existing CI and benchmarks

- Retain stable fuzz seeds and add bounded rotating-seed runs with minimized failures
  archived. Surface skip counts and unexpected skip growth.
- Report semantic partitions alongside aggregate coverage. Prioritize relevant
  assertions over raising coverage thresholds indiscriminately.
- Separate cold compile, hot cache, helper dispatch, array kernels and first JIT timing.
  Include small arrays, integer exactness and helper-heavy kernels, beyond float add.
- Add a backend support matrix covering JAX x64, Torch namespace/compat minimum,
  device and dtype limits, captures and fallback precision. CuPy and NumPy 1.x
  claims lack dedicated local/CI evidence in these reviews.
- Default dependency floors are exercised; optional backend floors are not covered
  by that job. Add evidence if those are intended support promises.
- Consider wheel/sdist smoke on pull requests, before release tags. No packaging
  defect was found; the current release isolation checks passed.
- Keep the lockfile, Make entry points, layer tests, canonical headers, golden AST
  comparison, hook/CI alignment, platform jobs and isolated publishing permissions.
  All four type checkers passed; measure unique benefit and runtime before changing
  that policy. Type checking does not replace semantic equivalence tests.

## Documentation changes worth making

The current separation of supported subset, semantics, API, architecture and
development pages is useful. Identifier/default checks and strict site builds
prevent structural drift, but do not prove behavioral claims.

Update the relevant pages with the fixes, rather than weakening the exactness
contract accidentally:

- Align bool/mixed arithmetic claims with actual guarded dispatch. The bool
  divergence row currently describes float64 casting while helpers typically use
  integer headroom; distinguish intended value semantics from representation.
- Explain branch/minmax promotion precision consequences, integer overflow/cast
  limits, division dtype/zero policies and floating library tolerances.
- State decorator, capture, helper, pinned/fallback and verification contracts clearly.
- Repair location/error examples, reserved-keyword/default preservation claims and
  namespace-variant body-preservation claims as their implementations are fixed.
- Provide a supported standalone-source export recipe. Executing copied source for
  `return x/y` without helper injection raises `NameError: _vec_arith is not defined`.
- Clarify the API table: `fallback=True` enables fallback as well as warning.
- Distinguish nested target/helper functions from rejected nested `def` statements;
  describe helper memoization with `(callee, protect, namespace)`.
- Execute complete copyable examples; add the missing `import inspect` to its
  quickstart snippet. Include error examples and helper/capture dependencies.
- Describe exactly what `make check` covers and which separate CI jobs add fuzz,
  backend, docs, smoke and floor evidence. A default gate need not run every expensive
  job, but its scope should be clear.

The documented count of 46 golden snapshots is correct: 45 canonical cases plus
one pinned variant. It is not a defect.

## Reconciliation decisions and withdrawn claims

Both reviewers used the documented divergences to narrow claims, rather than
combining every original assertion as a defect:

1. The huge-uint64/negative-int64 branch example is disclosed promotion behavior,
   with no common integer dtype. Even the positive-int64 variant is retained as a
   contract/enhancement concern, not an undisclosed P1 miscompile.
2. The original float64 `x + int(1e30)` script used a Python-int expectation for a
   float input. Scalar Python also returns `1e30`; that mismatch claim is withdrawn.
3. Subtracting positive `9223372036854775808` is not the AST for a signed-minimum
   literal. R24 uses the correct unary-minus spelling.
4. Documented verifier NaN-only exceptions and current-scalar verification after a
   capture mutation are policy/coherence limits, distinct from R19's blind spots/crash.
5. Captured-array backend locking is a portability limitation; automatic conversion
   is not prescribed without device/aliasing decisions.
6. Same-int64 overflow and mixed int/float min/max promotion are interpreted against
   their explicit divergence rules. They are not counted again as new P1s.
7. Protection code growth and an arbitrary 2x size test do not establish a correctness
   defect. The liveness counterexample in R06 does.
8. Public verification catches R04's supplied boolean mismatch; an exploratory probe
   label mentioning a missed comparison is not evidence that it passes verification.
9. Severity is P1/P2/P3, with narrow defaults, zero-sign, diagnostic and benchmark
   issues ranked proportionately. No P0 or security-critical claim is made.

Original scripts are retained byte-for-byte where copied, including invalid or
overbroad assertions. Their printed `DEFECT` totals are not the accepted finding
count. Use this report and the artifact interpretation notes to classify them.

## Original-report crosswalk

`A` refers to Reviewer A's F01–F18; `B` to Reviewer B's F1–F14 and subsections.

| Reconciled finding | Reviewer A | Reviewer B |
|---|---|---|
| R01 bool/unsigned unary | F01 | F1.1, F9; matrix additions |
| R02 mixed arithmetic | F02 | F1.1 |
| R03 runtime negative add/mul | — | F1.3 |
| R04 comparisons | F03 | independently reproduced during exchange |
| R05 decorators | — | F2 |
| R06 nested-loop protection | F04 | independently reproduced during exchange |
| R07 special-name binding | F05 | F4 |
| R08 cast/math boundaries | F06 | F8, F9 |
| R09 numeric composition/power | F07 | F9 |
| R10 strict argument normalization | F09 failed-fold subcase | F3, F9 |
| R11 scalar-only helper | F09 | independently reproduced during exchange |
| R12 min/max zero ties | F08 | independently reproduced during exchange |
| R13 integer zero remainder | — | F9 |
| R14 pin recapture | F10 | independently reproduced during exchange |
| R15 keyword binder | F11 | F4 |
| R16 effective defaults | F12 | independently reproduced during exchange |
| R17 source collisions | F13 | F5 |
| R18 fallback | F14 | F11 |
| R19 verification | F15 | F6; complex-pow reproduction during exchange |
| R20 fuzzer | F16 | F14; mutations reproduced during exchange |
| R21 benchmark | F17 | reviewed and agreed during exchange |
| R22 diagnostics | F18 | F10 |
| R23 concurrent helper compilation | initially a proposal; reproduced during exchange | F11; independent acceptance during exchange |
| R24 signed-minimum rejection | related F18 usability note | F8 |
| Contract/architecture/testing/docs proposals | architecture/testing/docs sections | F1.2, F7, F8, F11–F14, suggestions |

## Proposed implementation order

| Phase | Work | Acceptance evidence |
|---|---|---|
| 1. Strengthen oracles | R19/R20, plain-Python lane conversion, independent checker mutations | Integer/bool/sign/NaN/shape/error mutations fail; divergence classifications are explicit |
| 2. Repair wrong-result paths | R01–R08, including runtime helper negative results and decorator/binding policy | Minimal repros pass or reject clearly; dtype/source/control-flow properties on NumPy and strict |
| 3. Repair accepted combinations and artifacts | R09–R18, R23/R24; immutable artifact and source identity | Helper/scalar/strict, capture/default/keyword, concurrency, namespace and fallback precision checks |
| 4. Align tooling and claims | R21/R22, backend/range tables, executable examples, cold/hot benchmarks | Correct locations and timing intent; clear support/divergence tables; accurate standalone-source instructions |
| 5. Extend measured support | Optional floors, CuPy/GPU, guarded specialization, loop/protection optimization | Backend-specific semantic/performance evidence attached to each expanded promise |

Use D10 throughout: reject an unprovable combination until exact lowering exists.
Run `make check` and the relevant fuzz/backend/docs targets; update golden files only
for intentional emitted-source changes and review those diffs. Preserve the useful
existing engineering practices while replacing broad correctness claims with
discriminating evidence.
