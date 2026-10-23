# Repository review: array-vectorize

Reviewed on 2026-10-23 at commit `0e4bb6b64ae72ae369bbd61c921cf8673ec8d1a4`.

## Assessment

**CHANGES REQUESTED on semantic correctness.** The architecture is coherent and the
development infrastructure is strong, but accepted programs still produce incorrect
values, lose Python integer behavior, or fail on supported backends. Passing the
current gate does not establish the project's advertised exactness contract.

The highest-value next work is to close the gap between runtime dtype possibilities
and lowering decisions, strengthen the differential oracles, and preserve one frozen
compilation artifact across namespace variants. Expanding the supported syntax before
that work would multiply the combinations that need correction.

This report distinguishes reproduced defects from proposals. No implementation,
tests, golden files, dependency declarations, or documentation pages were changed.
Review probes were written under `/tmp`; the report includes minimal reproductions
so its evidence does not depend on those temporary files surviving.

## Scope and method

The review covered the compiler pipeline, runtime helpers, public API, source
inspection, verification and fallback, test organization and oracles, fuzzing,
benchmarks, user/developer documentation, Make targets, hooks, CI, packaging and
release scripts. It used the repository's
[semantics-review skill](../../.agents/skills/semantics-review/SKILL.md), including a fresh
independent compiler review, followed by rerunning the consolidated repros.

The strict semantic baseline is `docs/semantics.md`. In particular, NaN propagation
from `min`/`max`, branch dtype promotion, exceptional math lanes becoming IEEE values,
and documented unrepresentable-result approximations are not themselves defects.

Severity here means:

- **P1:** silent wrong results or a central exactness guarantee violated; prioritize
  before making broader correctness claims.
- **P2:** broken accepted combinations, public API contracts, or validation tools
  that can hide defects.
- **P3:** narrower diagnostic or development-quality defects.

### Validation actually performed

| Check | Result |
|---|---|
| `make check` | Passed: lint, four type checkers, 1,064 tests passed, 97 skipped; combined statement/branch coverage 96% |
| `make docs` | Passed strict documentation build |
| `make smoke` | Passed script, demo, and notebook execution |
| `make fuzz` | Passed all five seeds, 2,000 generated cases |
| Fuzz outcomes | 1,482 checked successfully, 2 rejected, 516 edge-skipped |
| `make backends` | Passed all 96 JAX/PyTorch compilation and backend tests in 115 seconds |
| `make lowest` | Passed: 1,064 tests passed, 97 skipped at direct dependency floors |
| `uv build --no-sources`, then `make package-smoke VERSION=0.1.0` | Passed wheel and sdist builds and isolated installed-artifact smoke tests |
| Enabled `test_compile_time` benchmark | Passed, but measured approximately 506 ns mean: see F17 |
| Independent numeric boundary matrix | 17,312 evaluated backend lane checks; 1,148 excluded scalar exceptions or values outside the matrix's chosen integer-result range |

The boundary matrix used NumPy and `array_api_strict`, bool/int64/uint64/float64,
Python scalar operands, signed zero, subnormals, infinities, NaNs, int64 limits,
values around `2**53`, and unsigned values above `2**63`. Additional probes covered
loops, helper calls, CSE, domain protection, API metadata, and rejected constructs.
The matrix's raw mismatches include documented divergences and repeated symptoms;
they are not a count of distinct defects.

The primary environment was Python 3.12.13, NumPy 2.5.3,
`array-api-compat` 1.15.0, `array-api-strict` 2.6.1, JAX 0.11.2, and
PyTorch 2.14.1+cpu. The isolated packaging smoke resolved NumPy 2.5.4.
The initial sandbox could not write uv's cache or reach dependency hosts; checks
were rerun successfully with the necessary execution access. That initial failure
was environmental, not a repository finding.

Windows, Python 3.13/3.14, CuPy/GPU behavior, and optional backend dependency floors
were not executed locally. CI configuration for the covered platforms was reviewed.
This is a broad audit with concrete counterexamples, not a proof over all programs.

## Project understanding and architecture

The package is a small eager source-to-source compiler, rather than an ndarray
interpreter or a replacement for `numpy.vectorize`:

1. `api.py` resolves options, canonical/helper/pinned caches, recursion detection,
   fallback, and public metadata.
2. `frontend/extract.py` obtains inspectable source and freezes closure/global
   values; `lambda_id.py` identifies lambdas using AST and code-object evidence.
3. `frontend/validate.py` checks structural subset membership; extraction and
   lowering perform additional semantic checks.
4. `lower/` assembles a stateful lowerer from inheritance layers for names, dtype
   facts, expressions, statements, loops, and helper calls. Branches become selects;
   constant-trip loops retain Python loops and explicit carried bindings.
5. Frozen dataclass IR nodes support constant folding, CSE, DCE, and a subsequent
   domain-protection pass. Loop-carried names require treatment beyond pure SSA.
6. `codegen/emitter.py` builds an AST, unparses it, and emits Array API operations.
   `emit.py` executes the source, injects helper functions and hidden defaults, and
   registers source with `linecache`.
7. The stdlib-only runtime leaf implements the exactness operations that cannot
   safely be left to generic backend promotion. Optional verification and fallback
   execute scalar Python lane by lane.

Several design choices are worth preserving:

- The pipeline/API split and injected helper-vectorizer callback avoid a compiler
  orchestration import cycle.
- The dependency-free IR and stdlib-only runtime boundaries have executable
  layering tests.
- AST-based emission avoids string-template escaping failures; `.source`,
  signatures, original-function markers, and docstrings make results inspectable.
- Integer true division has an explicit rounding algorithm and dedicated compiled
  backend tests, rather than relying on an early lossy float conversion.
- Tests cover several independent levels: white-box, behavior, strict backends,
  source goldens, differential properties, packaging, and infrastructure itself.
- CI checks multiple Python versions, Windows, dependency floors, optional
  compilation, docs, and fuzz seeds. Release builds are tested outside the checkout
  before a separate publishing job receives write permissions.

The main architectural weakness is semantic information split between the lowerer's
kind/maybe-bool/maybe-scalar facts and runtime dtype dispatch. An unknown parameter
is not evidence that a raw backend operator is safe. Several findings below are
consequences of treating it that way.

## Reproduced implementation and API defects

### Reproduction convention

Unless a snippet supplies its own imports, prepend the following, save the snippet
in a real `.py` file, and run it with `uv run python path/to/repro.py` from this
checkout. Repeat with `xp = np` and `xp = strict`. Source must be on disk because
`vectorize` uses `inspect.getsource`.

```python
import math
import numpy as np
import array_api_strict as strict
from array_vectorize import vectorize

xp = np  # repeat with strict
```

### F01 — P1: boolean operands bypass Python integer semantics

**Locations:** `lower/__init__.py:102`, `lower/expressions.py:71`,
`lower/expressions.py:330`, `lower/expressions.py:366`.

```python
def add(x, y):
    return x + y

def invert(x):
    return ~(x > 0)

print(vectorize(add)(xp.asarray([True]), xp.asarray([True])))
print(vectorize(invert)(xp.asarray([-1, 1])))
```

Python expects integer `[2]` and `[-1, -2]`. NumPy produces boolean `[True]`
and `[True, False]`; strict rejects the addition and produces the same incorrect
boolean inversion. Unary `+` and `-` on direct bool parameters also fail instead of
producing Python integers. `abs(bool)` and `round(bool)` have related backend/kind
problems.

Parameter initialization marks possible scalar values but not possible booleans.
Arithmetic dispatch consequently misses direct boolean inputs. Unary `+` and `~`
also lack complete Python bool-to-int handling even for known boolean expressions.

**Proposal:** make unknown input kinds conservatively include bool; centralize
Python unary and binary numeric dispatch, with fast paths for proven-safe dtypes.
Add direct-parameter, comparison-result, alias, branch-result, and helper-result
tests for each operator. Do not fix only `True + True`.

### F02 — P1: ordinary mixed integer arithmetic skips the exactness lattice

**Location:** `lower/expressions.py:366` and the raw-operator return at line 407.

```python
def subtract(x, y):
    return x - y

left = xp.asarray([2**63 + 1], dtype=xp.uint64)
right = xp.asarray([2**63 - 1], dtype=xp.int64)
print(vectorize(subtract)(left, right))
```

Python expects `[2]`, which fits in an integer array. NumPy returns float `[0.]`;
strict raises a dtype-promotion `TypeError`. Similar raw paths affect addition,
multiplication, remainder and floor division across mixed kinds. The elaborate
runtime helpers do not help when lowering never calls them.

This is outside the documented approximation for results that fit no integer dtype.
The chosen example has a small, exactly representable result.

**Proposal:** use the runtime lattice whenever operand kinds are unknown or their
combination requires Python promotion. Preserve simple float/same-dtype fast paths
inside the dispatcher or through guarded specialization. Add end-to-end tests
alongside existing helper unit tests.

### F03 — P1: comparisons round large integers before deciding equality/order

**Locations:** `lower/expressions.py:109`, `codegen/emitter.py` Compare emission.

```python
def equal(x):
    return x == 9007199254740993

print(equal(float(2**53)))
print(vectorize(equal)(xp.asarray([float(2**53)], dtype=xp.float64)))
```

Python returns `False`; both backends return `[True]`. Direct int64/float64 array
comparison also gives incorrect NumPy equality/order near the precision boundary,
while strict often rejects mixed array kinds.

**Proposal:** add exact mixed integer/float comparison logic rather than casting
both sides to float64. Cover equality and all order relations, scalar literals in
both positions, signed/unsigned arrays, infinities, NaNs, and integer bounds.

### F04 — P1: domain protection changes a live result through nested loops

**Location:** `optimize/protect_domains.py:167`.

```python
def nested(x):
    a = math.sqrt(x)
    s = 0.0
    for i in range(1):
        for j in range(1):
            s = s + a
    return a if x > 0 else s

x = xp.asarray([-1.0, 1.0])
with np.errstate(invalid="ignore"):
    print(vectorize(nested)(x))
    print(vectorize(nested, protect_domains=True)(x))
```

On both backends, unprotected output is `[NaN, 1.]`; protected output is `[0., 1.]`.
The unprotected NaN is the documented domain-error behavior. Protection promises
to preserve that result, but mistakenly considers `a` dead on the negative lane.

The pass marks immediate loop-body bindings fully live and skips reads inside
nested `Loop` nodes. Its separate bound-name traversal is recursive, making this
asymmetry easy to overlook.

**Proposal:** recursively collect reads from all nested statements and loop bounds
before propagating protection. Test protected/unprotected equivalence for nested
loops, with `isnan` and sign checks in addition to approximate values.

### F05 — P1: a rebound `range` is silently replaced with the builtin

**Locations:** `frontend/validate.py:validate_for`, `lower/loops.py:96`.

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

Python returns `[11, 12]`; both backends return `[2, 3]`. The frontend recognizes
the spelling `range` without establishing its binding. The tuple-returning helper
is outside the supported subset, so strict compilation should reject this function
rather than silently ignore the helper.

**Proposal:** resolve the callable to the actual builtin and reject global,
nonlocal, parameter, or local shadowing. Use binding identity for other special
syntax/call recognizers as well.

### F06 — P1: `int` and `math.trunc` corrupt representable uint64 values

**Locations:** `frontend/tables.py:69`, `frontend/tables.py:89`,
`lower/expressions.py:205`.

```python
def subject(x):
    return int(x)

print(vectorize(subject)(xp.asarray([2**63 + 1], dtype=xp.uint64)))
```

Python expects positive `9223372036854775809`; both backends produce negative
`-9223372036854775807`. Replacing `int` with `math.trunc` has the same problem.
The original value already fits uint64; the unconditional int64 cast destroys it.

**Proposal:** preserve integer operands when the Python operation is an integer
identity; use checked conversion for floating operands. For results outside every
available integer dtype, define and enforce a rejection/divergence policy instead
of relying on an unchecked signed cast.

### F07 — P2: integer-returning builtins cannot safely compose with integer operators

**Locations:** `frontend/tables.py:49`, `frontend/tables.py:74`,
`lower/expressions.py:223`.

```python
def subject(x):
    return math.floor(x) & 1

print(vectorize(subject)(xp.asarray([1.5])))
```

Python expects `[1]`; both backends raise `TypeError` because `xp.floor` returns
floating arrays. `math.ceil(x) & 1` and `round(x) & 1` fail similarly. Merely
comparing standalone rounded values with `allclose` misses this difference.

A related accepted combination, `def reciprocal(x): return x ** -1`, raises
`ValueError` for int64 `[2]` on both backends instead of returning `[0.5]`:
`lower/expressions.py:403` only casts bases already proven integer, leaving direct
unknown parameters unhandled.

**Proposal:** track Python result kinds separately from backend result dtypes,
then materialize the required integer or floating representation when composing
operations. Test compositions and runtime input kinds, not only individual calls.

### F08 — P2: `min`/`max` lose Python's first-operand signed-zero tie behavior

**Location:** `runtime/minmax.py:40` through its pairwise fold.

```python
def subject(x, y):
    return math.copysign(1.0, min(x, y))

x = xp.asarray([0.0, -0.0])
y = xp.asarray([-0.0, 0.0])
print(vectorize(subject)(x, y))
```

Python expects `[1., -1.]`; both tested backends return `[-1., 1.]`.
`max` has the same tie issue. The documented min/max NaN divergence does not permit
changing the sign of finite zero ties.

**Proposal:** use a stable comparison/select fold that retains the first equal
operand while explicitly preserving the documented NaN policy. Cover both operand
orders and multiargument ties.

### F09 — P2: scalar-only helper calls fail inside valid array calls

**Location:** `lower/helpers.py:35` and line 43.

```python
def helper(y):
    return y * y

def subject(x):
    return helper(2) + x

print(vectorize(subject)(xp.asarray([1, 2])))
```

Expected `[5, 6]`; both backends raise `TypeError` because the separately compiled
helper tries to detect a namespace from the raw scalar `2`. The caller already has
a valid namespace, but the helper does not inherit that runtime context.

**Proposal:** propagate the caller's runtime namespace into helper invocation,
or inline/fold helpers where that is sound. Preserve scalar helper semantics
without depending on which arguments happen to remain array-valued.

There is another scalar/backend boundary hole: `return math.sqrt(-1.0) + x`
produces the documented NaN on NumPy but raises `TypeError` on strict. When constant
folding declines an exceptional math call, a raw float reaches `xp.sqrt`.
See `lower/sanitize.py:76`, `optimize/constfold.py` failed-call folding. Ensure that
unfolded calls also receive valid Array API arguments.

### F10 — P2: namespace pinning recompiles and changes frozen captures

**Location:** `api.py:103` through line 111.

```python
SCALE = 2

def subject(x):
    return x * SCALE

v = vectorize(subject)
SCALE = 3
print(v(np.asarray([1])))
print(v.with_namespace(np)(np.asarray([1])))
```

Observed `[2]` followed by `[3]`. The API and implementation comments say namespace
variants change only the namespace binding and do not need reverification because
the body is unchanged. In fact, `_vectorize_strict(original, ...)` repeats source
extraction and capture, so mutable global/nonlocal bindings and helper environments
can change the body. Protected compilations sharing one pin cache can also obscure
which frozen environment a variant represents.

**Proposal:** retain an immutable compilation artifact containing resolved source,
captures, helper artifacts and options. Generate namespace variants from that
artifact, and key caches by its identity. If recompilation is intentionally exposed,
give it a separate explicit API and document the recapture behavior.

### F11 — P2: reserved parameter renaming breaks advertised keyword calls

**Locations:** `ir/ssa.py:28`, `ir/ssa.py:62`,
`codegen/emitter.py:_build_signature`, `emit.py:78`.

```python
import inspect

def subject(array_namespace):
    return array_namespace + 1

v = vectorize(subject)
print(inspect.signature(v))
print(v(array_namespace=np.asarray([2])))
```

The advertised signature is `(array_namespace)`, but the call raises
`TypeError: ... unexpected keyword argument 'array_namespace'`. Positional calls
work. `hasattr` has the same reserved-name treatment. Setting `__signature__`
changes introspection, not the generated function's actual argument binder.

**Proposal:** preserve public parameter spellings and allocate collision-free
aliases for imports/internal names. Test every reserved/internal base name through
keyword calls, including keyword-only parameters.

### F12 — P2: actual defaults can disagree with generated defaults and signature

**Locations:** `frontend/extract.py:59`, `codegen/emitter.py:_build_signature`,
`emit.py:78`.

```python
import inspect

def subject(x, y=2):
    return x + y

subject.__defaults__ = (7,)
v = vectorize(subject)
print(subject(1), v(np.asarray([1])), inspect.signature(v))
```

Observed scalar `8`, vectorized `[3]`, advertised signature `(x, y=7)`.
Defaults are read from source AST while the attached signature comes from the
current function object. This occurs even when mutation precedes the first
compilation, so it is unrelated to the documented first-compilation freeze.

**Proposal:** obtain defaults from the resolved callable's effective signature,
validate those values against the supported policy, and use the same frozen data
for generation and metadata. Add analogous `__kwdefaults__` coverage. A conservative
explicit rejection when source and callable disagree is safer than silent mismatch.

### F13 — P2: compiling a namesake overwrites source inspection and traceback text

**Location:** `emit.py:46` and line 66.

```python
import inspect

def first_factory():
    def subject(x):
        return x + 1
    return subject

def second_factory():
    def subject(x):
        return x * 9
    return subject

a = vectorize(first_factory())
before = inspect.getsource(a)
b = vectorize(second_factory())
print(before != inspect.getsource(a))
print(inspect.getsource(a) == inspect.getsource(b))
```

Both comparisons print `True`: inspecting `a` now shows `b`'s code. `.source` still
contains `a`'s actual generated code. The filename key is only
`<array_vectorize:subject>`, so same-named functions, lambdas, and namespace/protect
variants overwrite one another. Tracebacks can therefore display the wrong line.

**Proposal:** use a unique or content-addressed generated filename, including the
relevant generated source identity. Test simultaneous variants and namesakes;
current tests pin a fixed filename but do not test collisions.

### F14 — P2: the correctness fallback rounds float64 values to float32 on Torch

**Location:** `fallback.py:63`; related test at
`tests/backends/test_jax_torch_compilation.py:test_fallback_with_torch_tensors`.

```python
import warnings
import torch
import array_api_compat.torch as txp

def subject(x):
    while x < 0:
        x += 1
    return x

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    v = vectorize(subject, fallback=True, namespace=txp)
x = torch.tensor([1.0000000001], dtype=torch.float64)
print(x.item(), v(x).item(), v(x).dtype)
```

Observed `1.0000000001 1.0 torch.float32`. This changes values, not merely output
dtype. The test explicitly accepts backend default float32 output but uses only
exactly representable sample values, so it misses the precision loss.

**Proposal:** construct output with a policy preserving Python scalar precision
and integer range. Define empty-output dtype/device behavior as part of that policy.
Test float64 values off the float32 grid and large integers, on multiple backends.

Also, pinned fallback wrappers still reject all-scalar calls at `fallback.py:42`,
while the namespace API broadly promises that pinning makes such calls legal.
Support that case or explicitly qualify the documentation for fallback wrappers.

## Defects in verification and development tools

### F15 — P2: `verify=` cannot establish integer, kind, or signed-zero exactness

**Location:** `verify.py:87` through line 102.

Both sides are cast to float64 and compared with a relative/absolute tolerance.
This erases integer distinctions above `2**53`, permits integer errors within a
relative tolerance even below that limit, treats booleans like integers, and
equates positive and negative zero.

The verifier's seam was tested directly to isolate it from compiler behavior:

```python
from array_vectorize.verify import verify_match

def identity(x):
    return x

verify_match(lambda x: x + 1, identity,
             (xp.asarray([2**53], dtype=xp.int64),))
verify_match(lambda x: -x, identity,
             (xp.asarray([0.0], dtype=xp.float64),))
```

Both calls incorrectly pass on both backends. These are synthetic wrong compiled
results used to test the checker, not additional compiler miscompile claims.
The existing verifier does correctly catch F03's boolean mismatch when those
example inputs are explicitly supplied.

**Proposal:** compare bool/integer values exactly without a lossy common cast,
check result kinds under an explicit divergence policy, and compare zero sign
where the operation's contract requires it. Keep tolerances only for justified
floating backend differences. Include deliberate one-bit, one-unit, sign, kind,
shape and NaN mutations in the verifier's tests. Remove or make lazy the redundant
`xp.asarray(expected_values)` conversion used only for the error message.

### F16 — P2: the fuzzer can accept wrong results and suppress genuine failures

**Locations:** `fuzz.py:175`, `fuzz.py:196`, `fuzz.py:201`, `fuzz.py:226`.

Three weaknesses were exercised directly:

1. Replacing the compiled result of `return 2.0` with an all-`True` boolean array
   yields `_check_case(...) == "ok"`. If either result is boolean, both are coerced
   to bool, erasing the value and kind mismatch.
2. Replacing that compiled callable with one that unconditionally raises
   `TypeError("genuine runtime failure")` yields `"edge-skip"`. The catch does not
   establish that a documented dead-lane divergence caused the error.
3. `fuzz.main(["--seed", "0", "--cases", "1", "--seconds", "0"])` returns success
   after zero cases, despite `--cases` being documented as overriding the time
   budget. `0` also becomes an unlimited case count through `args.cases or inf`.

The current value pool is seven small finite floats, and the oracle primarily uses
`allclose`. It cannot exercise the dtype combinations behind F01–F03/F06. A
passing run with 516/2,000 edge-skips gives weaker evidence than a passing run of
2,000 checked programs, although the CLI does print the outcome counts.

**Proposal:** fail on kind mismatches; classify each permitted divergence narrowly;
fail unexpected exceptions; add typed grammar/input strategies and signed-zero
checks; validate CLI bounds and make count/time termination match its documented
meaning. Preserve minimized failing source, inputs, backend and seed as artifacts.
Add mutation tests for all three demonstrated blind spots.

### F17 — P2: the compile benchmark measures canonical-cache hits

**Locations:** `tests/perf/test_bench.py:99`, `api.py` canonical-cache hit path.

`benchmark(vectorize, subject)` repeatedly passes the same function object. The
first invocation compiles; subsequent invocations return the memoized result. The
enabled benchmark reported about **506 ns mean** across 707 rounds, evidence of
API/cache overhead rather than compilation of a branch-and-loop function.

**Proposal:** split cold compilation and hot lookup benchmarks. For cold runs,
construct a fresh function object outside the timed interval or use a deliberately
uncached compiler entry point; specify whether helper caches are warm or cold.
Validate compile-stage invocation counts so future caching changes cannot silently
change what the benchmark measures.

### F18 — P3: diagnostics do not meet the documented location/collection guarantee

**Locations:** `frontend/info.py:FunctionInfo`, `frontend/validate.py:305`,
`lower/env.py:106`, `errors.py:Diagnostic.render`, `frontend/extract.py:51`.

For an unsupported `while` in a function defined at file line 40, the actual output
reports `line 2, col 4` with no filename. Extraction errors such as a negative
default (`y=-1`) have an empty diagnostics list; many lowering errors contain only
a relative line number. `FunctionInfo.filename` is captured but no original source
start offset is retained for this reporting path.

`docs/api.md`, `docs/quickstart.md`, and the architecture promise file/line/column
locations and collected violations. Collection currently applies chiefly to the
structural validator, not every rejection. The negative-default example is also a
useful usability gap: syntactically signed literals are rejected as non-literals.

**Proposal:** store original file and block offset, use one structured diagnostic
constructor across extraction/validation/lowering, and specify exactly which
violations can be collected. Test nested/decorated definitions, Unicode columns,
and defaults. Either support signed scalar defaults or document their exclusion.

## Testing assessment and missed opportunities

### Make semantic coverage visible, not just line coverage

The 96% coverage and 1,064 passing default tests coexist with simple wrong-result
programs above. This is a coverage-distribution problem, not a request for more
tests indiscriminately. Existing runtime tests exercise sophisticated helper paths
while ordinary unknown-parameter operators can bypass those helpers entirely.

`tests/differential/test_scalar_equivalence.py:36` also invokes scalar functions
with NumPy scalar elements for many properties. Those can inherit NumPy promotion,
overflow and operator behavior instead of Python scalar behavior. Some targeted
integer tests explicitly convert to `int`; the shared oracle should do so uniformly.
Its `allclose` comparisons also omit kind and signed-zero checks.

**Proposal:** create a shared contract-aware lane oracle that converts backend
elements to Python scalars; checks exact bool/integer values, NaN status and zero
sign; and applies named, narrow divergence rules. Use it for Hypothesis, verification
and fuzzing where their policies overlap. Keep independent tests of the oracle so
a shared checker defect cannot invalidate every layer at once.

Generate a coverage matrix over operation × input kind × operand source × backend
× control-flow context. Operand sources should include parameters, literals,
captures, assignments, helper results, and loop-carried values. Prioritize boundary
partitions around precision and representation changes, not only random magnitudes.

### Compare compiled backends against scalar Python as well as eager output

Most JAX/PyTorch compilation tests intentionally compare compiled results with the
same generated function running eagerly. That is a useful tracing test, but shared
compiler mistakes pass on both sides. Retain those tests and add scalar-oracle
checks for representative semantic boundaries. Document JAX's x64 requirement,
Torch namespace choice, supported devices/dtypes, and actual CuPy test coverage in
a backend support matrix. CuPy is advertised but has no dedicated CI job here.

The dependency-floor job covers the default dev group. Optional JAX/Torch floors
are not validated by that job; add an optional-floor job if those floors are intended
as supported compatibility promises. Keep the existing documented special minimum
for detected-mode Torch compilation visible in the matrix.

### Test API combinations and failure classifications

Add focused pairwise coverage for namespace × fallback × verify × protection ×
cache state × capture mutation. Examples in this review show why testing each
option alone is insufficient. Include scalar-only helper arguments, zero-dimensional
and empty arrays, broadcasting, keyword-only calls, device preservation, and
simultaneously live generated variants.

Use adversarial checker tests: intentionally wrong values, signs, kinds, shapes,
exceptions and generated names should cause the corresponding verification tool
to fail. The existing test named `test_verify_detects_mismatch` in
`tests/features/test_verify.py` actually checks successful two-dimensional
verification; rename it and add an actual failure assertion for that purpose.

## Architecture and implementation enhancement proposals

These are proposals, not additional reproduced wrong-result findings.

| Proposal | Benefit and implementation boundary | Priority / estimated size |
|---|---|---|
| Immutable compilation artifact | Freeze source, effective signature, captures, lowered IR, helper graph and options once; derive pins without re-extraction. Addresses F10/F12 and clarifies cache identity. | High / medium-large |
| Explicit abstract input-kind lattice | Distinguish proven kinds from sets of possible runtime kinds; carry Python result kind separately from backend dtype and raw-scalar status. Addresses several numeric findings at their shared cause. | High / large |
| Semantic operation registry | Extend the current name/arity tables with result-kind, domain, scalar conversion and exceptional-behavior metadata; reduce duplicated special cases across lowering, folding and verification. | High / medium |
| IR invariant validator | Check definition/use order, loop scope, emitted-name uniqueness, scalar/backend argument requirements, and result-kind invariants after each pass in tests/debug mode. | Medium / medium |
| Cache lifecycle policy | Global canonical/helper/pin caches are explicitly unbounded and strongly retain original functions, closures, captured arrays and namespaces. A weak-reference probe confirmed discarded originals remain live. Define session scope, clearing and observability; consider weak ownership carefully because values reference keys. | Medium / medium |
| Concurrency policy | `_ACTIVE_HELPERS` is process-global compilation state. Consider per-compilation or context-local recursion tracking and synchronized cache publication; add concurrent-compilation tests if thread-safe use is supported. This risk was identified by inspection, not reproduced here. | Medium / medium |
| Loop-aware optimization regions | CSE currently skips an entire program if it contains any loop (`optimize/cse.py:47`). Optimize safe straight-line regions first, then consider loop-local CSE with explicit dominance/liveness rules. Benchmark before expanding scope. | Low / medium-large |
| Exportable source bundle | `.source` can reference injected runtime/user helpers and hidden captures. Add an export API including dependencies/bindings, or qualify paste-and-run claims. | Medium / medium |

The lowerer's inheritance layers make responsibilities navigable, but its shared
mutable state and repeated loop-body lowering require careful snapshot discipline.
Do not start with a wholesale rewrite. Add invariants and explicit semantic facts,
then replace fragile special cases incrementally with regression-backed changes.

Runtime value-dependent dtype choices deserve a documented performance model:
calls such as `bool(xp.all(...))` can synchronize devices and behave differently
under tracing. The existing exact integer-division documentation acknowledges its
cost well. Extend that treatment to the other runtime lattice paths, using measured
eager/JIT/GPU benchmarks when those configurations are available.

## Documentation assessment

The documentation has a useful separation: supported subset, semantics, API,
architecture and development. Executable identifier/default checks and a strict
site build help prevent structural drift. Those checks do not establish behavioral
accuracy, and several current claims are stronger than implementation evidence:

- Exact mixed arithmetic and boolean handling are advertised as general guarantees;
  F01/F02 show missing dispatch paths.
- Namespace variants are described as preserving the compiled body; F10 shows
  recapture instead.
- Signature/default/keyword preservation needs F11/F12 fixed.
- `docs/quickstart.md` says generated source can be pasted into another codebase.
  Executing `vectorize(lambda-like-ratio).source` for `return x / y` in fresh globals
  leaves `_vec_arith` undefined; calling it raises `NameError`. Runtime/helper
  injection and closure defaults need an export recipe or an explicit limitation.
- `docs/semantics.md` says boolean arithmetic casts operands to float64, while
  runtime arithmetic normally chooses integer headroom for bools. Align this row
  with the actual intended lattice after the dispatch defects are fixed.
- The top-level fallback description is a correctness escape hatch, but precision
  and all-scalar pin behavior need an explicit contract (F14).
- The quickstart's `inspect.getsource` example omits `import inspect`; add it to the
  snippet or make the sequence executable as a documented whole.

**Proposal:** execute selected complete documentation examples, including copyable
generated source, unsupported examples and error text. Add a backend limitations
table and a short explanation of output shape/kind for constant returns, empty
inputs, captured arrays and fallback. Explain verification tolerance and scope so
users do not interpret a passing example check as proof of exactness.

## Development, CI and release assessment

Keep the existing Make entry points, lockfile, source/golden separation, canonical
license checker, hook/CI command alignment, and installed-artifact release smoke.
Those mechanisms worked in this review. The repository also tests its own hook,
release, license, portability and agent-layout conventions, which makes its process
less dependent on undocumented maintainer knowledge.

The following improvements would make those practices yield stronger evidence:

1. **Archive review/fuzz failures in CI.** Upload minimized source, inputs, backend
   versions, seed and classified failure. Run a bounded rotating-seed job in addition
   to the five stable seeds; retain deterministic seeds for reproducibility.
2. **Separate benchmark intentions.** Distinguish compile time, cache lookup,
   runtime dispatch, array kernel time, and first JIT compilation. Add integer
   exactness/helper-heavy representative kernels to the performance report; the
   current simple float-add gate cannot characterize those costs.
3. **Surface skip and semantic coverage.** Record why tests/fuzz cases skip and
   make unexpected skip growth visible. Report boundary partitions covered alongside
   aggregate branch coverage. The aggregate threshold currently allows important
   modules such as `lower/env.py` and `optimize/constfold.py` below 90% individually;
   use targeted semantic assertions before simply increasing numeric thresholds.
4. **Exercise package builds before release tags.** Installed wheel/sdist smoke
   currently runs in release flow. A pull-request packaging smoke would catch
   distribution problems before the release process, although no such packaging
   defect was found here.
5. **Measure the cost of four type checkers.** All passed, and their wider test
   scopes have value. Record runtime and unique defects caught before changing
   that policy; this review does not support removing them. They validate typing,
   not the scalar/array semantic equivalence that needs stronger oracles.

No release was published or tagged during the review. Local build outputs are
ignored artifacts under `dist/`.

## Recommended implementation order

| Phase | Work | Acceptance evidence |
|---|---|---|
| 1. Make correctness checks discriminating | F15/F16; shared Python-scalar oracle; preserve separate oracle mutation tests | Wrong integer, bool, zero-sign and exception mutations fail; count-limited fuzz runs execute the requested cases |
| 2. Repair known wrong-result paths | F01–F08, especially dtype dispatch, comparisons, protection recursion and binding resolution | All minimal repros pass on NumPy/strict; targeted scalar differential properties; appropriate golden review |
| 3. Repair call and artifact semantics | F09–F14; frozen compilation artifact and collision-free source identity | Helper/scalar calls, capture mutation, actual keyword binding/defaults, precision and simultaneous variant tests |
| 4. Align public claims and tooling | F17/F18, executable docs, support matrix, cold/hot benchmark split | Accurate diagnostic locations, documented limitations, benchmark stage counts, strict docs build |
| 5. Broaden supported configurations | Optional-floor testing, CuPy/GPU evidence, measured optimization work | Backend-specific correctness and performance results attached to each new claim |

For each compiler fix, keep the project rule: reject an unprovable combination
until it has an exact lowering. Run `make check` plus the relevant fuzz/backend/docs
targets. Regenerate source goldens only where the intended emitted code changes,
and inspect those diffs. The existing green gate is a useful baseline; the new
counterexamples must become discriminating regressions before it can guard the
advertised contract.

## Preserved review artifacts

The scripts, results, validation logs, benchmark JSON, and exact smoke-tested wheel
and sdist are archived under
[review-artifacts/2026-10-23](review-artifacts/2026-10-23/README.md).
The archive includes reproduction instructions and a SHA-256 manifest. Probe source
bodies are preserved with canonical repository license headers prepended.

The following temporary files were executed during this review. The minimal
snippets above and archived scripts are the durable reproduction references; the
paths below identify their original review-session locations.

- `/tmp/semantics_review_repros.py`: consolidated compiler findings on both backends.
- `/tmp/probe_matrix.py`, `/tmp/probe_matrix_functions.py`: boundary matrix and subjects.
- `/tmp/probe_helper_literal.py`: scalar-only helper call.
- `/tmp/probe_project_api_review.py`: source collision, keyword/default/capture cases,
  verifier mutations, pinned fallback, cache retention.
- `/tmp/probe_project_quality_review.py`: fuzz oracle mutations and CLI termination.
- `/tmp/probe_project_additional_review.py`: Torch fallback precision, pasted source,
  default rejection, and diagnostic locations.
- `/tmp/array-vectorize-review-extra-checks.log`: docs, smoke, fuzz, optional backends.
- `/tmp/array-vectorize-review-lowest.log`: dependency-floor suite.
- `/tmp/array-vectorize-review-packaging.log`: builds and installed-artifact smoke.
- `/tmp/array-vectorize-review-benchmark.json`: enabled compile benchmark output.
