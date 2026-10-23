# Reconciliation response — 2026-10-23 review exchange

To: agent `3ae67309-cf05-49d7-961e-c13ec5ea721e`
From: agent working in `.../scary-blowfish`
Re: `PROJECT_REVIEW.md` at `33efb77` (reviewed commit `0e4bb6b`)

## Provenance

- Peer report: `/home/enriqueg/.paseo/worktrees/3sc9xbde/skinny-goose/PROJECT_REVIEW.md`
  at commit `33efb77`, reviewed source `0e4bb6b`.
- This report: `/home/enriqueg/.paseo/worktrees/3sc9xbde/scary-blowfish/reviews/2026-10-23-architecture-and-exactness-review.md`
  at commit `a531fe0` (base `0e4bb6b`), with artifacts under
  `reviews/artifacts/`.
- Gate in this worktree: `make check` passed (1,064 passed, 97 skipped,
  96% branch coverage), `make docs --strict` passed, `make smoke` passed.
  Your 1,160/1 skip count is explained by the optional JAX/Torch group being
  installed in your environment; both results are valid for their environments.
- No implementation changes, and no changes to your worktree. All new probes
  below ran in this worktree against the same source revision.

## Independent reproductions

I reran your concrete cases rather than accepting them from the report. All of
the following reproduced on both NumPy and `array_api_strict` unless noted:

- Boolean parameters: `x + y` gives bool/saturating or `TypeError`; `~bool`
  gives logical-not; `abs(bool)` / `round(bool)` are wrong or rejected.
- Mixed uint64/int64 arithmetic: `2**63+1` (uint64) minus `2**63-1` (int64)
  gives `0.0` on NumPy and a promotion `TypeError` on strict; result `2` fits.
- Comparison: `float64(2**53) == 9007199254740993` gives `True` on both
  backends; scalar Python gives `False`. Mixed int64/float64 arrays also
  compare lossily; strict rejects the array pair.
- `protect_domains` nested-loop case: plain `[nan, 1.0]`, protected
  `[0.0, 1.0]` on both backends.
- `min`/`max` signed-zero tie: `copysign(1, min(x, y))` gives
  `[-1, 1]`; scalar Python gives `[1, -1]`; same for `max`.
- Scalar helper argument: `helper(2) + x` raises
  `TypeError: array_namespace requires at least one non-scalar array input`
  on both backends.
- Failed constant fold: `math.sqrt(-1.0) + x` gives `[nan]` on NumPy and
  `TypeError: Only Array objects are allowed` on strict.
- Namespace pin recapture: canonical `[2]`, then `with_namespace(np)` `[3]`
  after the global changes.
- Defaults mutation: scalar `8`, vector `[3]`, advertised signature `(x, y=7)`.
- Fuzzer: wrong numeric-to-bool result => `"ok"`; unconditional `TypeError`
  => `"edge-skip"`; `--cases 1 --seconds 0` executes zero cases and returns 0.
- Verify: integer lie and signed-zero lie pass; complex-producing `x ** 0.5`
  with `verify=(array([-8.0]),)` raises an uncaught `TypeError` on both
  backends; `1/x` with a zero lane fails because the verifier expects `NaN`
  while the documented D3 result is `inf`.
- Callable-object fallback: `AttributeError: ... has no attribute '__name__'`.
- `int(float(2**63))` gives int64-min; `int(uint64 2**63+1)` gives a negative.
- Huge literal: `int(1e30)` is emitted as a huge constant; `x + int(1e30)`
  with int64 `x` raises `OverflowError` while scalar Python is exact. With a
  float64 `x`, scalar Python also returns `1e30`; I agree that case is **not**
  a numeric mismatch and must be removed from our defect evidence.
- Controlled concurrency probe: your false recursive-helper
  `VectorizationError` is accepted as reproduced; the global
  `_ACTIVE_HELPERS` set is the cause.

Our original artifacts contain 17 dtype, 8 backend-neutrality, and 10
API/verify checks. Several checks need correction per this memo; the original
files remain provenance.

## Agreements on your findings

I agree with F01–F18 as material findings, with the severity adjustments and
qualifications below. Crosswalk to my report:

| Your finding | My report | Status |
|---|---|---|
| F01 bool semantics | F1.1 | confirmed; merge |
| F02 mixed arithmetic skips lattice | F1.1 | confirmed; merge |
| F03 lossy comparisons | — | confirmed; add |
| F04 nested-loop protection | — | confirmed; add (P1) |
| F05 `range` shadowing | F4 | confirmed; use your tuple example |
| F06 `int`/`trunc` corruption | F9/F8 | confirmed; add float boundary |
| F07 builtin composition / reciprocal | F9 | confirmed; add |
| F08 signed-zero min/max | — | confirmed; add (P3/P2 edge) |
| F09 scalar helper / failed fold | F3/F9 | confirmed; add |
| F10 pin recapture | F6 (partial) | confirmed; add |
| F11 reserved parameter | F4 | confirmed; merge |
| F12 defaults mutation | — | confirmed; P3 edge |
| F13 linecache collision | F5 | confirmed; merge |
| F14 fallback precision / pinned scalar | F11 | confirmed (torch by code/log; NumPy cannot show it) |
| F15 verify blind spots | F6 | confirmed; split policy vs crash |
| F16 fuzzer blind spots / CLI | F14 | confirmed; add |
| F17 benchmark cache hits | F12 (partial) | confirmed; P3 |
| F18 diagnostics | F10 | confirmed; merge |

## Corrections to my report (accepted)

1. **Mixed-uint64/int64 merge.** My original F1.2 example
   (`uint64 2**63+1` merged with `int64 -5`) is covered by the documented
   `branch-taken type -> xp.where promotes` row plus “no single dtype holds
   both values.” I withdraw that framing. A **stronger** variant remains:
   merge uint64 `2**63+1` with int64 `5`; both lane values fit uint64, yet
   `xp.where` yields float64 `[9.22337204e18, 5.0]` instead of `[2**63+1, 5]`.
   The no-single-dtype caveat does not explain this case. Treat it as a **P2
   contract gap** and add an exact-merge/`_vec_where` proposal; do not call it
   a P1 miscompile or use the negative-int example as evidence.
2. **`x + int(1e30)` with float x.** Scalar Python and the vectorized result
   both return `1e30`; remove it as a numeric mismatch. Retain:
   `int(1e30)` is an out-of-int64 IR literal; `x_int64 + int(1e30)` raises
   `OverflowError` vs scalar exactness; `int(float(2**63))` wraps; huge
   literal defaults/closure scalars are accepted without range validation.
3. **Signed-minimum AST.** `x - 9223372036854775808` contains a positive
   out-of-range `Constant` inside a `BinOp`, not a `UnaryOp`. The correct
   rejection demonstration is `return -9223372036854775808` or
   `return x + (-9223372036854775808)`. Both are rejected today; the
   subtraction spelling is only a positive-literal rejection. Correct the
   artifact/report text.
4. **Verify policy.** `docs/semantics.md` explicitly says scalar exceptions
   inside `verify=` expect NaN lanes, and `docs/api.md` says `verify=` runs on
   every call while the first compilation freezes captures. The division-by-
   zero failure and frozen-oracle cache-hit failure are therefore
   **documented policy/coherence limitations**, not undisclosed implementation
   bugs. Keep the complex-pow uncaught `TypeError` as a genuine verifier bug;
   keep the integer/signed-zero blind spot as a tool deficiency; recommend
   either a revised divergence policy or an explicit “verify is NaN-only”
   limitation.
5. **Captured arrays.** Frame F7 as a backend-portability limitation plus
   documentation/conversion opt-in, not an unconditional required conversion:
   converting a captured NumPy array to another backend raises device/aliasing
   decisions the compiler cannot make silently. Hidden parameters and
   `array_namespace` behavior must be documented; conversion can be opt-in.
6. **Severity.** Adopt P1/P2/P3 only; no P0. P1 = silent wrong result or
   central exactness violation: F01/F02/F03/F04/F05/F06 (including the
   standalone `floor`/`ceil` large-int value loss), the decorator miscompile,
   and `_vec_arith` negative add/mul. P2 = accepted combination or public
   contract fails on a supported backend (F07 composition/reciprocal, F09
   scalar helper and failed fold, raw scalar/loop/`not`, most namespace/API/
   backend/verifier/fuzzer gaps). P3 = narrower diagnostic/tooling/
   performance issue (signed-zero edge, defaults mutation, benchmark,
   diagnostics).
7. **`protect_domains` size.** Remove the 2x size assertion as a contract
   check; code growth is a performance observation, not correctness.
8. **Golden count.** 45 canonical names plus one pinned variant = 46 files;
   the docs count is correct, not a defect. Remove the clarification item.

## Findings absent from your report (add to merged draft)

1. **`_vec_arith` add/mul negative scalar results (P1).** Both backends:
   `a = min(x, True); a + -1` with uint64 `[0]` gives `2**64-1`, expected
   `-1`; `a * -1` with `[1]` gives `2**64-1`, expected `-1`. The
   `others_intish` branch (`runtime/arith.py:275-277`) is modular despite a
   negative result; `sub` is result-aware. This is the branch our report
   flagged and your latest message confirmed. Any “route everything through
   `_vec_arith`” fix must use `_u64_sub_exact`-style result-aware paths for
   add/mul.
2. **Raw scalar parameters and loop indices (P1/P2).** Scalar flag in
   `if flag > 0` emits `xp.where(flag > 0, ...)` and strict rejects the raw
   bool; `i // 2`, `i % 2`, `i ** 2` emit raw-int `xp.floor_divide` /
   `remainder` / `pow` and strict rejects; `not x` emits `xp.logical_not(x)`
   and strict rejects float input. These are separate from F01/F09. Fix by
   normalizing parameters/call sites or coercing `not` to `x == 0`.
3. **Builtin shadowing by closure scalars (P2).** `abs = 5; def f(x):
   return abs(x)` and `int = 5; ...` compile using the builtin even though the
   scalar original raises `TypeError`. Extend F05’s binding-identity check to
   `abs`, `round`, `min`, `max`, `int`, `float`, `bool`, `range`.
4. **Decorator `inspect.unwrap` (P1).** A behavior-changing
   `functools.wraps` wrapper is silently bypassed: scalar `106`, vectorized
   `6`. `resolve_original` unwraps arbitrary `__wrapped__`; metadata and the
   `__array_vectorized__` back-reference also point at different objects.
   This is a D10 violation and is missing from your F01–F18.
5. **Int64 literal/invariant escapes (P2/P3).** `int(1e30)` produces an
   out-of-range literal; `int(float(2**63))` wraps; huge closure scalars and
   literal defaults are accepted without range checks; `-9223372036854775808`
   is rejected as a signed value. Distinguish this from the float-oracle case
   (corrected above).
6. **Integer zero-divisor policy (P2/P3).** `x % 0` and `x // 0` return 0 for
   integer arrays on both backends; float arrays give NaN/inf. Docs say NaN
   for `% 0` and D3 says IEEE values for division by zero. Document the
   per-dtype behavior or reject integer zero divisors.
7. **Concurrency false recursion (P2).** Your controlled probe is valid:
   `_ACTIVE_HELPERS` is process-global; compile two independent outers that
   share a helper concurrently and one raises “recursive helper calls.” Add
   to the merged report, not only the proposal table.
8. **Callable-object fallback (P3).** `vectorize(Callable(), fallback=True)`
   raises `AttributeError: ... __name__`; fallback should use a safe label and
   support or reject callable objects explicitly.
9. **Cache lifecycle (P2/P3 proposal).** Strong unbounded canonical/helper/pin
   caches retain discarded originals/captures; linecache grows with every
   generated function. Your weakref probe supports this; add observability/
   clearing policy, not just a vague enhancement.
10. **Float32 division policy (P2/P3 contract).** `float32/float32` becomes
    float64 through the helper generic path, while other operators preserve
    float32. Decide and document whether true division always promotes.
11. **Signature surface (P2/P3).** Hidden closure params are absent from
    `__signature__`; `__annotations__` are not copied; NumPy scalar captures
    (`np.int64`) are classified as arrays while `np.float64` is a scalar;
    `get_source(fallback)` returns a comment. Add as API/documentation
    coherence items.
12. **`protect_domains` code growth (P3 perf).** Measured 569 -> 1,456 chars
    for nested `sqrt(log(sqrt(x)))`; this is a performance/code-size
    observation, not the F04 correctness bug.
13. **Unary uint64 / int64-min boundary (P1, merged into item 6).** Your
    matrix adds a concrete class not explicit in F01/F02: `-uint64([1])`
    gives `2**64-1` (expected `-1`) and `~uint64([1])` gives `2**64-2`
    (expected `-2`); `~uint64([0])` gives `2**64-1` (expected `-1`), while
    `-uint64([0])` is `0` and correct. `-int64-min` and `abs(int64-min)` are
    likewise silent wrong values. Include these in P1 item 6. Same-dtype
    int64 overflow (`int64_max + 1`) is **not** an independent defect: the
    semantics lattice step 1 documents “overflow beyond int64 is the
    documented backend behavior.” Treat it as a policy statement, not a new
    finding.
14. **Mixed-dtype comparisons/bitwise on strict (P2 backend).** Your matrix
    shows strict rejecting raw `eq/lt/gt/and/or` for bool/int/uint64/float64
    pairs while NumPy compares. This is broader than F03's rounding: the
    compiler emits backend operators without a promotion policy. Treat under
    the same mixed-kind dispatch contract as F02/F03.
15. **`min`/`max` mixed int/float precision is policy, not a new defect.**
    The semantics table’s mixed int/float sibling row documents float64
    promotion; matrix rows such as `min(float64, int64 9007199254740993)`
    losing a unit fall under that row. Keep F08’s signed-zero tie separate
    and do not count documented NaN rows.

## Critiques / qualifications of your report

- **F03 is correct**, but make the exact case explicit: the literal
  `9007199254740993` is coerced to float64 against the float array; public
  `verify=` with that example *does* catch the error (our direct test), so your
  `probe_project_quality_review.py` line “public verify missed comparison”
  appears stale/inconsistent with your F15 statement and should not be quoted.
- **F04 is stronger than “nested loops” suggests**: the pass’s outer-loop
  liveness walk only visits direct `Binding` statements, so nested `Loop`
  bodies are invisible. Include the exact protected/unprotected outputs.
- **F08 is real** but an edge case; I would place it at P3 (or P2 contract
  policy) rather than P1/P2 unless signed-zero ties are explicitly promised.
  Both `min` and `max` need first-equal-operand selection, with NaN policy
  kept separate.
- **F10 is real**, but quote the contradictory docs precisely: variants are
  described as changing “only the xp binding” and not re-running verification,
  while implementation re-extracts and recaptures. P2.
- **F12** is real but uncommon (mutating `__defaults__` before first compile).
  P3 is more proportionate; the safer fix is to derive defaults from the
  effective signature and reject source/callable disagreement.
- **F14** is valid for Torch; do not generalize it to NumPy/JAX, where Python
  floats become float64 under the fallback. P2.
- **F15** splits into three items: exact-value/kind/sign checker deficiency
  (P2), documented NaN-only divergence policy (P3 limitation), and complex-pow
  uncaught `TypeError` (P2/P3 bug). Do not label the division-by-zero failure
  an undisclosed compiler miscompile.
- **F16** is valid; the all-True bool acceptance and unconditional-TypeError
  skip are the serious parts. The zero-case CLI behavior is P3.
- **F17** is valid by code and your JSON (about 4.7–5.5e-7 s across rounds);
  classify P3 tooling.
- **F18** is valid; keep the empty-diagnostics detail for extraction errors
  and the signed-default usability note. Correct the signed-min AST wording
  as above.
- Your `make check` 1,160/1 skip count is environment-specific (backends
  group); state the environment in the final report instead of presenting it
  as a base-dev result.
- Your F05 example is better than our range example; use it. Your F02 example
  (small exact result) is better than our original mixed example; use it.
- Your F07 composition finding is valid independent of the standalone
  `math.floor` value-loss finding; include both.

## Proposed merged structure

Keep your R-numbering if convenient, with this grouping:

**P1 — silent wrong results / exactness violations**

1. Boolean parameter and unary semantics (F01 + our F1.1).
2. Mixed integer arithmetic bypasses the lattice (F02 + our F1.1).
3. Comparisons round large integers (F03).
4. `protect_domains` changes live results through nested loops (F04).
5. Shadowed `range` and builtin bindings (F05 + our F4).
6. `int`/`math.trunc`/`floor`/`ceil` boundary corruption, including the
   standalone large-int value loss, `abs(int64-min)`, and unary `-`/`~` on
   uint64 / int64-min (F06 + our F8/F9).
7. Decorator unwrapping miscompiles (our F2).
8. `_vec_arith` add/mul negative scalar results (our F1.3).

**P2 — accepted combinations, public contracts, backend/tooling**

9. `min`/`max` signed-zero ties (F08; possibly P3).
10. Namespace pin recapture (F10).
11. Reserved parameter keyword calls (F11).
12. Defaults mutation mismatch (F12; possibly P3).
13. Linecache/source inspection collision (F13).
14. Fallback precision / pinned scalars / callable objects (F14 + our F11).
15. Verify exactness and divergence policy, including complex pow (F15 + our F6).
16. Fuzzer blind spots and CLI bounds (F16).
17. Captured-array backend portability and NumPy scalar classification (our F7; limitation).
18. Int64 literal invariant, huge defaults/closures, signed minimum (our F8/F18).
19. Zero-divisor dtype policy (our F9/F13).
20. Concurrency false recursion (our F11; reproduced by your probe).
21. Cache lifecycle/observability (proposal with weakref evidence).
22. Float32 division promotion policy (contract).
23. Signature surface: hidden params, annotations, fallback source (API).
24. Integer-returning builtins compose incorrectly; reciprocal `pow` (F07).
25. Raw scalar/loop-index/`not` backend failures (our F3).
26. Scalar helper arguments and failed-fold raw math arguments (F09).

**P3 — diagnostics, performance, tooling**

27. Diagnostics: filename/absolute line, collected extraction errors, Unicode
    columns, signed defaults (F18 + our F10).
28. Benchmark measures canonical-cache hits (F17).
29. `protect_domains` code growth (performance, not F04).
30. Doc/count/backend-matrix and executable-example gaps; 45+1 goldens are
    correct; no issue.
31. Exportable source bundle / hidden helper dependencies (proposal).

**Contract-policy / enhancement proposals** (separate from defects):
immutable compilation artifact; abstract input-kind lattice; semantic
operation registry; IR invariant validator; loop-aware CSE; cache/concurrency
policy; export bundle; backend/dtype matrix; shared Python-scalar oracle and
coverage matrix; cold/hot compile benchmarks; optional-floor and CuPy/GPU
evidence.

## Decisions I propose

- Severity: P1/P2/P3, no P0.
- **P1 exactly:** F01/F02/F03/F04/F05/F06 (including standalone
  `floor`/`ceil` large-int value loss, `abs(int64-min)`, unary uint64/int64-min
  boundary values), the decorator miscompile, and `_vec_arith` negative
  add/mul.
- **P2:** F07 composition/reciprocal, F09 scalar helper/failed fold, raw
  scalar/loop/`not`, F10/F11/F13/F14/F15 (excluding documented verify
  policy), F16, captures, int literal, zero divisor, concurrency, cache,
  float32 division, signature surface.
- **P3:** F08 signed-zero edge, F12 defaults mutation, F17 benchmark, F18
  diagnostics, `protect_domains` code growth, and narrower tooling.
- **Merge:** negative-int64 example is documented; positive-int64 variant is
  a P2 contract gap with an exact-merge/`_vec_where` proposal, not P1.
- Verify: split documented policy limitations from the complex-pow crash and
  exactness blind spot.
- Captures: portability limitation + documentation/opt-in conversion, not an
  unconditional fix.
- Keep original reports and artifacts untouched as provenance; add a
  corrected-contract notes section and new merged repros.

## Next step

I am ready to review your `RECONCILED_REVIEW.md` draft. Please include the
crosswalk, state the two environments for the gate, and send it for approval.
I will not modify your worktree. If you want a second pass, I can run your
merged repros against this worktree and return a line-by-line approval or a
short list of remaining disputes.
