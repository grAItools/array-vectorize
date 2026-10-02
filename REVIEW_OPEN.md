# Open review findings — round 3 (reviewer agent 94e52ea7, reviewed commit baf7137)

Verdict: REQUEST_CHANGES. 456 tests passed; perf test failed at 3.1x vs its
10x gate; fresh mypy reports 3 errors in `_optimize.py`. All findings below
were reproduced by the reviewer under `/tmp` with the project venv.

Repros assume `math`, `numpy as np`, `vectorize` imported.

## 1. CRITICAL — Literal substitution retains invalid facts across loops
`src/vectorizer/_lower.py:249`, `src/vectorizer/_lower.py:477`

```python
def f(x):
    a = x
    for i in range(0):
        a = 4.0
    return math.sqrt(a)
```
At `[9., 16.]` returns scalar `2.0` instead of `[3., 4.]`. The literal from
the unexecuted body is recorded and substituted into `sqrt`, removing the
input dependency. Also with executed loops: `a = 4.0` then conditionally
`a = 9.0` inside `range(1)` — loop-carried sync updates the binding without
invalidating its literal fact, so `sqrt(a)` folds to the stale constant.

Planned fix: in `lower_for`, snapshot `name_literals` before body lowering;
after the loop, restore the pre-loop facts and drop entries for every name
bound in the body (incl. nested for-targets via `_body_bound_names(
include_for_targets=True)`) and every carried name.

## 2. CRITICAL — Numeric-to-boolean loop transitions still miscompile
`src/vectorizer/_lower.py:448`, `src/vectorizer/_lower.py:458`

```python
def f(x):
    b = 1
    a = x
    for i in range(2):
        a = b + b
        b = x > 0
    return a
```
At `[1.]` output is `[True]` instead of `[2]`. Body is lowered while `b` is
classified int; kind updates after body lowering do not retrofit the needed
conversion. The `* 1` mitigation never reaches this op.

Planned fix: needs joint redesign with finding 6 (see below) — the `* 1`
intify is broken on strict backends anyway.

## 3. CRITICAL — Builtin sanitization converts exact integers to floats
`src/vectorizer/_lower.py:642`, `src/vectorizer/_lower.py:753`

```python
def f(x):
    return x + abs(9007199254740993)
```
At integer input `[0]` returns `9007199254740992.0` (loses 1). Also
`return abs(3) & x` raises TypeError because `abs(3)` becomes `3.0`.

Planned fix: split the sanitizer — math-table calls keep float-forcing
(consistent with scalar `math.*` semantics), but polymorphic builtins
(`abs`, `round`, `min`, `max`) must use a soft mode: literal →
`asarray(literal)` (exact, no dtype cast); array-valued refs untouched.

## 4. CRITICAL — Lambda matching merges signed-zero constants
`src/vectorizer/_extract.py:103`, `src/vectorizer/_extract.py:137`

```python
f1, f2 = (
    lambda x: math.copysign(x, 0.0),
    lambda x: math.copysign(x, -0.0),
)
```
`vectorize(f2)(np.array([1.]))` returns `[1.]`; scalar returns `-1.0`.
Tuple equality on `co_consts` merges `0.0` and `-0.0` (`==` is True), both
candidates match, and the fallback picks the wrong one.

Planned fix: sign-aware normalization of constants on BOTH sides of the
comparison (`float` zeros keyed with `math.copysign(1, c)`); recurse into
nested const tuples. AST `Constant(-0.0)` preserves the sign, so the AST
side already distinguishes them once compared with a sign-aware key.

## 5. MAJOR — Domain protection creates forward/self references
`src/vectorizer/_optimize.py:539`

```python
def f(x):
    if x < 0:
        return 0.0
    y = math.sqrt(x)
    return y if y > 1 else 0.0
```
Protected evaluation at `[-1., 4.]` raises UnboundLocalError for `y`
(expected `[0., 2.]`). The propagated liveness ctx (contains `y > 1`) is
inserted into the expression defining `y`; likewise a conditional variable
assigned after a partial-function binding is referenced before assignment.
Backward liveness propagation must respect where guard expressions can
actually be evaluated.

Planned fix: at rewrite time, check the clamp condition's free names; if a
name is not bound strictly before the statement being rewritten (or is the
statement's own target), fall back: drop that use's ctx to the empty
intersection / treat the partial call as fully live (no clamp). Alternative
considered: SSA-side def-before-use analysis of ctx refs.

## 6. MAJOR — `* 1` does not convert booleans on the strict backend
`src/vectorizer/_lower.py:731`

```python
import array_api_strict as xp

def f(x):
    return (x > 0) + (x > 0)

vectorize(f)(xp.asarray([1.]))
```
Raises `TypeError: Only numeric dtypes are allowed in __mul__`. Generated
code uses `(x > 0) * 1`, but strict boolean arrays do not support numeric
multiplication.

Planned fix: kind-aware intify — when the operand's static kind is 'bool',
emit `astype(asarray(x), int64)` (strict-safe) instead of `* 1`; keep `* 1`
only where the operand is provably numeric, or replace entirely. For
loop-kind mixing (finding 2) the phi merge itself usually promotes
bool|int to int — verify what the generated source actually does for the
finding-2 repro before redesigning.

## 7. MAJOR — Computed scalar math arguments remain unsanitized
`src/vectorizer/_lower.py:757`

```python
def f(x, y=4.0):
    return x + math.sqrt(y + 0.0)
```
With `array_api_strict.asarray([1.])` raises TypeError (raw float reaches
`xp.sqrt`). Sanitizer wraps unknown Refs but leaves compound expressions
unchanged; a compound expr can still evaluate to a Python scalar (param
defaults are raw scalars at runtime).

Planned fix (root cause): preamble-wrap every parameter in
`xp.asarray(param)` right after `array_namespace(...)` — arrays are no-ops,
raw default scalars become 0-d arrays. This also fixes finding 8 and lets
Ref sanitization shrink to literals only. Expect mass golden updates.

## 8. MAJOR — Scalar arguments to casts still crash
`src/vectorizer/_lower.py:651`

```python
def f(x, y=3.0):
    return x + int(y)
```
Raises `AttributeError: 'float' object has no attribute 'astype'` instead
of `[4.]`. Builtin casts emit `xp.astype(y, ...)` directly; `math.trunc`
same path.

Planned fix: covered by the finding-7 preamble wrap (params become arrays
before any use). Literal args keep const-folding.

## 9. MINOR — Committed code fails strict mypy
`src/vectorizer/_optimize.py:462`, `:496`, `:502`

- Missing type arguments for `tuple` in `_clamp_dead`.
- Unused `type: ignore` in `or_live` (the `_unrecorded` sentinel).
- Incompatible `tuple[object, Node]` argument passed to `Logical`.

Planned fix: annotate properly (`Node | tuple[...]`), remove the stale
ignore, type the `Logical` operands as `tuple[Node, Node]`.

## 10. Perf gate — timing-sensitive failure
`tests/test_perf.py` (10x gate) failed at 3.1x on the reviewer's loaded
machine; passed locally at ~40x. Not counted as a correctness finding by
the reviewer, but it failed their run. Consider marking it
`@pytest.mark.timing`/skippable under load, or documenting in the review
instructions that timing tests may fail on loaded machines and are not
gate-blocking.

## Status
- None of the round-3 fixes have been started yet (this file written
  immediately after receiving the round-3 verdict).
- Suggested order: 9 (mypy, trivial) → 7+8 (preamble wrap) → 3 (soft
  sanitize) → 4 (signed zeros) → 5 (ctx availability) → 6+2 (bool intify
  redesign) → 1 (literal invalidation) → regressions in
  tests/test_review_round3.py → goldens → make check + fuzz → commit/push
  → round-4 review request.
