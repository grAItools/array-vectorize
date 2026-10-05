# Restructuring Plan: `vectorizer`

Status: **executed — phases 0–6 complete (commits 89c5cb5..HEAD); gates green throughout.**
Base: commit `d24ceaf` (includes the upstream quality pass `ea24c71`,
lambda-identification fix `5ebff7d`, and its regression coverage `d24ceaf`).
Gates at base: ruff, mypy (strict), ~650 tests, 95% branch coverage, fuzzer
seeds 42/7/123/999/2024, new CI fuzz/docs jobs, `make bench`.

## 1. Scope and constraints

Restructure the package and the test suite as if designed from scratch,
removing the structural debt of the incremental (26-round adversarial review)
development process. Two hard constraints:

1. **Python floor 3.12–3.14**: `requires-python = ">=3.12"`; drop 3.10/3.11;
   modernize the code to 3.12+ idioms (see §7).
2. **Zero user-visible behavior change** during the migration: public API
   (`vectorize`, `get_source`, `VectorizationError`, `python -m vectorizer.fuzz`),
   generated code output, documented semantics, and all quality gates stay
   intact at every phase boundary.

## 2. Baseline snapshot (at `d24ceaf`)

- Source: `src/vectorizer/` — 13 modules, ~4,000 LOC, all private (`_`-prefixed)
  except `fuzz.py`; `py.typed` ships types.
- Tests: `tests/` — ~6,100 LOC, ~505 test functions (~650 with parametrization),
  95% branch coverage gate.
- Docs: Zensical site in `docs/` (6 pages), marimo notebooks in
  `examples/notebooks/`, `make docs` / `make bench` / `make fuzz` /
  `make notebook` targets; CI runs check (matrix 3.10–3.13), fuzz, and docs
  jobs.
- Test scaffolding is deduplicated in `tests/support.py` (`make_fn`,
  `make_module`, `TMPDIR`) — landed in `ea24c71` (−570 lines).
- `conftest.py` registers a derandomized Hypothesis CI profile (300 examples)
  and pytest runs with `filterwarnings = ["error", ...]`, `strict_markers`,
  `pytest-benchmark` (disabled by default), `pytest-timeout`.

## 3. Upstream changes already incorporated into this plan

The three upstream commits changed the landscape this plan was drafted
against:

1. **Test helper dedup (`tests/support.py`)** — the mechanical half of the
   test-consolidation proposal is done. What remains is *regrouping* the
   test files by theme (§6), not writing shared helpers. `tests/support.py`
   becomes the seed of the target test layout and gains the strict-backend
   fixture and corpus loader.
2. **Lambda identification rewrite in `_extract.py`** (~+70 lines) — `_find_target`
   now uses a two-strategy matcher (recompile the target's module and pair
   code objects with AST nodes by line/order; fall back to isolated-probe
   bytecode comparison). This subsystem (`_find_target`,
   `_lambdas_in_source_order`, `_iter_code_objects`, `_code_key`,
   `_defaults_match`) is now ~150 lines with its own regression suite
   (`tests/test_extract.py`, 25 tests) and is promoted to its own module in
   the target design (§5: `frontend/lambda_id.py`).
3. **New infra to keep green**: `tests/test_bench.py` imports private members
   of `vectorizer._runtime` (`_ARITH_OPS`, `_vec_arith`, `_vec_minmax`) — the
   runtime split (§9 phase 3) must update these imports in the same commit.
   `docs/architecture.md` documents the module layout and pipeline — every
   source-moving phase now also updates the docs page in the same commit.
4. **Found docs bug to fix immediately**: the `docs/architecture.md` mermaid
   diagram shows `optimize` running **before** `lower`; the actual pipeline is
   `lower → optimize`. Fix on landing, independently of the restructure.

## 4. Diagnosis — incremental-development debt

### 4.1 Source (`src/vectorizer/`)

**a) Two lifecycles conflated in `_runtime.py` (487 lines).** It mixes
generation-time machinery (`compile_vectorized`: compile/exec/linecache/
wrapper attributes) with the call-time runtime library (`_vec_arith`,
`_vec_minmax`, `_u64_sub_exact`, dtype analysis) that is injected into
generated modules and executes on the user's backend at call time.

**b) `_lower.py` is a god module (1,107 lines — 2× the next largest).**
`_Lowerer` accreted at least five responsibilities: statement lowering +
environment tracking (`definite`/`maybe`/`deferred`), branch merging,
loop-phi machinery with fixed-point re-lowering, expression lowering, and a
de facto type-inference system smeared across five parallel dicts
(`name_kinds`, `name_literals`, `_scalar_names`, `_maybe_bool_names`,
`_carried_assign_kinds`) — the direct residue of review rounds 10–26, where
each round bolted on another tracking dimension instead of designing a kind
lattice.

**c) Layering violations.** The claimed pipeline is
`extract → validate → lower → optimize → codegen → compile`, but imports show:

- `_lower` → `_optimize` for private tree utilities (`_children`,
  `_const_fold_expr`) — generic IR algebra lives in the wrong layer;
- `_lower` → `_runtime` for private callables (`_vec_arith`, `_vec_minmax`,
  `_ARITH_OPS`) — lowering crosses the generation/call-time boundary via
  hardcoded string names (`"_vec_minmax"` literals in `lower_function`);
- `_codegen` → `_lower` for `LoweredFunction` — the pipeline's shared data
  type is defined mid-pipeline, forcing a forward dependency.

**d) `LoweredFunction` is a kitchen-sink DTO** (IR program + codegen metadata
+ runtime-injection metadata), which is what causes (c).

**e) Smaller scatter:** `protect_domains` (a user-facing semantic feature)
buried inside `_optimize.py` as a pseudo-pass; `_is_array`/`_py_scalar`
duplicated in `_fallback.py` and `_verify.py`; `__init__.py` owning pipeline
orchestration plus a global never-evicted `_HELPER_CACHE` keyed by function
objects.

### 4.2 Tests (`tests/`)

**a) 25 `test_review_roundN.py` files (221 tests, ~2,300 lines) organized by
discovery date, not behavior.** The underlying themes are clear once named:
bool/minmax arithmetic exactness (~80 tests), uint64 arithmetic exactness
(~60), loops/SSA/CSE (~15), protect_domains (~10), verify (~8),
helpers/caching (~8), extraction/lambda identification (~8 + the new 25-test
suite), fallback (~3). Names like `round3_pipeline_smoke` and
`min_max_roundtrip_prior_findings` encode history, not specification. Provenance
already lives in git history.

**b) Process-artifact organization elsewhere:** `test_m4.py` (milestone),
`test_coverage_gaps.py` (coverage target), `test_smoke.py` vs `scripts/smoke.py`
vs `examples/demo.py` (three overlapping smokes).

**c) Repeated corpus loading** via `spec_from_file_location` in
`test_golden.py`, `test_behavior.py`, `test_differential.py`,
`test_fuzz_smoke.py`, `test_bench.py`; strict-backend tests each do inline
`import array_api_strict as xps` instead of a shared fixture/parametrization.

## 5. Target source design

Rules decided up front rather than accreted:

1. **One dependency direction**: `frontend → ir ← lower → optimize → codegen
   → emit`, with `runtime` (call-time) and `api` at the edges. Nothing
   imports backwards or sideways.
2. **The IR owns its own algebra.** Tree walking/rewriting (`children`,
   `rewrite`) belongs to `ir`, not to a consumer.
3. **Kind inference is an explicit component with one data structure**, not
   five parallel dicts.
4. **Generation-time and call-time code never share a module.** Runtime
   helpers are referenced through a registry keyed by stable names.

```
src/vectorizer/
├── __init__.py          # public API re-export only (vectorize, get_source,
│                        #   VectorizationError) — ~10 lines
├── api.py               # vectorize(): option handling, fallback decision,
│                        #   helper cache + recursion detection (from __init__)
├── errors.py            # VectorizationError, Diagnostic            (was _errors)
├── pipeline.py          # compile_function(): extract→validate→lower→optimize→
│                        #   codegen→emit (was __init__._vectorize_strict)
│
├── frontend/            # "understand the scalar function"
│   ├── extract.py       #   source extraction, closure capture   (was _extract)
│   ├── lambda_id.py     #   lambda identification subsystem (~150 lines,
│   │                    #     grew in 5ebff7d; own regression suite)
│   ├── info.py          #   FunctionInfo, Param — dependency-free types
│   ├── validate.py      #   subset validation + diagnostics      (was _validate)
│   └── tables.py        #   name→xp mapping tables               (was _builtins)
│
├── ir/                  # the intermediate representation
│   ├── nodes.py         #   dataclasses, Node/Stmt unions, is_bool  (from _ir)
│   ├── walk.py          #   children/rewrite/map — generic algebra  (from _optimize)
│   └── ssa.py           #   SSAEnv, generated_name, reserved names  (from _ir)
│
├── lower/               # AST → IR  (the god module, split by concern)
│   ├── __init__.py      #   lower_function() entry point
│   ├── types.py         #   LoweredFunction DTO — shared with codegen/emit,
│   │                    #     breaking the codegen→lower dependency
│   ├── env.py           #   lowering state: SSA defs, branch env, deferred returns
│   ├── statements.py    #   assign / if / merge machinery
│   ├── loops.py         #   for-loops, phis, fixed-point kind widening
│   ├── expressions.py   #   binops, compares, boolops, calls, casts
│   ├── kinds.py         #   ★ the explicit kind lattice:
│   │                    #     @dataclass VarInfo: kind|None, literal, scalar,
│   │                    #     maybe_bool — replaces the five parallel dicts
│   ├── sanitize.py      #   _sanitize_xp_arg / _as_float_arg / literal resolution
│   └── helpers.py       #   user-helper vectorization + runtime-helper name
│                        #     allocation (names resolved via runtime.registry,
│                        #     no hardcoded "_vec_minmax" strings)
│
├── optimize/            # IR → IR passes
│   ├── constfold.py     #   (from _optimize)
│   ├── cse.py
│   ├── dce.py
│   └── protect_domains.py  # promoted out of _optimize — a semantic
│                           # user-flagged feature, not a generic pass
│
├── codegen/             # IR → readable Python source
│   └── emitter.py       #   (was _codegen; imports lower/types.py + ir only)
│
├── runtime/             # ★ CALL-TIME library injected into generated modules
│   ├── registry.py      #   RUNTIME_HELPERS: {"vec_arith": fn, "vec_minmax": fn},
│   │                    #     ArithOp ids — the single seam between lowering
│   │                    #     decisions and emitted helper calls
│   ├── dtype.py         #   _describe_dtype / _fits_dtype / _minmax_bound_kind
│   ├── arith.py         #   _vec_arith, _u64_sub_exact
│   └── minmax.py        #   _vec_minmax
│
├── emit.py              # GENERATION-TIME loading: compile/exec/linecache/
│                        #   wrapper attributes (compile_vectorized, was _runtime)
├── verify.py            # differential verification              (was _verify)
├── fallback.py          # element-loop fallback                  (was _fallback)
├── compat.py            # shared _is_array/_py_scalar (dedupe verify+fallback)
└── fuzz.py              # stays: `python -m vectorizer.fuzz` is a documented
                         #   public CLI; py.typed stays
```

Import DAG (all arrows point down):

```
api ──► pipeline ──► frontend ──► ir
              │         │
              ▼         ▼
            lower ◄── (ir.walk, ir.ssa)
              │
              ▼
          optimize ──► codegen ──► emit

runtime (leaf; imported only via registry by pipeline/lowering, and by emit
         for injection)          verify, fallback, compat (leaves under api)
```

Consequences:

- `lower` no longer imports `_optimize` or private runtime callables; it
  consumes `ir.walk` and `runtime.registry`.
- `codegen` no longer imports `lower`; both consume `lower/types.py`.
- `__init__.py` becomes a thin re-export; the global helper cache moves to
  `api.py` (weakref/eviction story is an optional follow-up, out of scope
  here).
- `tests/test_bench.py`'s imports move to `vectorizer.runtime` in the same
  commit as the runtime split.

## 6. Target test design

Principle: **organize by behavior and level, never by discovery date,
milestone, or coverage target.** `tests/support.py` (upstream) is kept and
extended — it becomes the home of the strict-backend fixture and the shared
corpus loader (killing the repeated `spec_from_file_location` blocks).

```
tests/
├── conftest.py          # + strict backend fixture + corpus loader helper
├── support.py           # make_fn/make_module (upstream, unchanged)
├── corpus.py            # golden corpus (unchanged)
├── unit/                # white-box, per source module
│   ├── test_extract.py      # + lambda_id tests (25 upstream tests)
│   ├── test_validate.py     # + test_validate_branches.py merged in
│   │                        # + 3.12–3.14 new-syntax rejection tests (§7.1)
│   ├── test_ir.py           # + ssa/walk unit tests
│   ├── test_lower.py
│   ├── test_optimize.py     # constfold/cse/dce as separate classes
│   ├── test_codegen.py
│   └── test_runtime.py      # ★ new: direct unit tests of runtime/arith,
│                            #   runtime/minmax, runtime/dtype (today only
│                            #   covered end-to-end; test_bench.py covers
│                            #   performance, not semantics)
├── behavior/            # black-box end-to-end semantics on NumPy
│   ├── test_arithmetic_dtypes.py   # bool/int/uint64 exactness matrix —
│   │                                #   absorbs ~140 minmax/arith round tests
│   ├── test_control_flow.py        # branches, early returns, merges, SSA rebinds
│   ├── test_loops.py               # loops, phis, zero-trip, nested, kind mixing
│   ├── test_calls_math_casts.py
│   ├── test_closures_and_helpers.py
│   └── test_divergences.py         # T8 documented divergences
├── features/                # public-option behaviors
│   ├── test_fallback.py            # fallback tests scattered across rounds
│   ├── test_protect_domains.py     # from test_m4.py + round tests
│   └── test_verify.py              # verify= option
├── backends/
│   └── test_array_api_strict.py    # all *_strict tests, via the shared fixture
├── differential/test_differential.py   # unchanged (Hypothesis)
├── golden/                  # test_golden.py + cases/ (unchanged — ast.dump
│                            #   comparison is refactor-safe by design)
├── test_inspectability.py   # source/getsource/signature/idempotence
├── test_perf.py             # slow-marked (unchanged)
└── test_bench.py            # slow+benchmark-marked (unchanged, imports updated)
```

Dissolution map: `test_review_round1..25.py` → dissolved into `behavior/`,
`features/`, `backends/` by theme (tests moved verbatim, only imports
updated); `test_m4.py`, `test_coverage_gaps.py`, `test_behavior.py`,
`test_composition.py`, `test_smoke.py` → dissolved into
`behavior/`/`features/`; `scripts/smoke.py` and `examples/demo.py` stay
(runnable docs, not tests).

## 7. Python 3.12–3.14 floor + modernization

### 7.1 The version bump changes the input language (correctness obligation)

`vectorize()` ingests user source code. On a 3.12–3.14 interpreter, user
functions can contain 3.11–3.14 syntax, and the project contract is
*"rejected with a precise diagnostic — never silently miscompiled."* The
validator must explicitly cover (currently all missing or generic-fallback):

| New syntax | Node | Required handling |
|---|---|---|
| `except*` (3.11) | `ast.TryStar` | explicit rejection message |
| `type X = ...` (3.12, PEP 695) | `ast.TypeAlias` | explicit rejection message |
| `def f[T](x)` (3.12, PEP 695) | `FunctionDef.type_params` | reject non-empty `type_params` on the extracted function and nested defs (today the body silently vectorizes if `T` is unreferenced) |
| t-strings (3.14, PEP 750) | `ast.TemplateStr` | explicit rejection message; **guard with `getattr(ast, "TemplateStr", None)`** — the one legitimate version branch, because the AST grammar itself varies by interpreter |
| `except A, B:` (3.14, PEP 758) | parses to `ast.Try` | already rejected; nothing to do |

Each new rejection test is version-parametrized so the `TemplateStr` case
runs on 3.14 and the guard path on 3.12/3.13 — that is what the CI matrix
is for.

### 7.2 Floor-bump checklist (mechanical)

- `pyproject.toml`: `requires-python = ">=3.12"`; classifiers `3.12/3.13/3.14`
  (drop `3.10`, `3.11`); `[tool.ruff] target-version = "py312"`;
  `[tool.mypy] python_version = "3.12"`.
- Dev pin bumps, with reasons: `numpy>=1.26` (first release with cp312
  wheels — the current `>=1.24` floor cannot install on 3.12), `mypy>=1.13`
  (PEP 695 `type` statement support), `ruff>=0.8` (pre-commit already at
  v0.16.10, satisfied).
- `.github/workflows/ci.yml`: check-job matrix `["3.12", "3.13", "3.14"]`;
  the fuzz and docs jobs already pin `3.12` — keep as the floor.
- `README.md` and `docs/index.md`: "Requires Python >= 3.10" → ">= 3.12".
- Dev environments on 3.10 (including this worktree's sandbox) must
  provision a 3.12+ interpreter before phase 0 can verify anything.

### 7.3 Modernization inventory (mapped to the target layout)

Unconditional (floor 3.12):

| Feature | Where | Value |
|---|---|---|
| PEP 695 `type` aliases | `type Node = Literal \| Ref \| ...`, `type Stmt = ...`, `type HelperVectorizer = ...`, `type Kind = Literal["int","float","bool"]`, `type ParamKind = Literal["posonly","arg","kwonly"]` | `Literal`-typed aliases turn `dict[str, str]` kind tracking into `dict[str, Kind]` — mypy strict then catches invalid kind/op strings statically |
| `match`/`case` + `typing.assert_never` | every isinstance dispatch chain: `_gen_expr` (12 branches), `validate_expr`/`validate_stmt`, `_children`/`_rewrite` (→ `ir/walk.py`), `_numeric_kind`, `lower_expr`, `_fold_*` (~115 isinstance sites total) | current `raise TypeError("unexpected IR node ...")` fallbacks become static exhaustiveness checks: adding an IR node becomes a mypy error at every dispatch site instead of a runtime surprise. Highest-value modernization |
| `@dataclass(frozen=True, slots=True)` | IR nodes, `Diagnostic`, `Param`, `LoweredFunction` | free memory/speed for structural hashing in CSE; field-based equality unchanged |
| `enum.IntEnum` for arith op ids | `runtime/registry.py` — replaces the `_ARITH_OPS` tuple + `.index()` arithmetic in lowering | kills the magic-number convention between lower and runtime (see gotcha 7.4.1) |
| PEP 701 f-strings, `typing.override`, `itertools.batched` | incidental | cosmetic |

Deliberately kept:

- `from __future__ import annotations` stays (13 files). `_ir.py` annotates
  fields with `Node`/`Stmt` before those unions exist; PEP 649 only removes
  the need when the floor is 3.14. Valid and harmless on all three versions.

Floor-gated (3.13/3.14-only) — recommended **not** to use now:

- `typing.TypeIs` (3.13) for `_is_array`/`_is_u64` narrowing,
  `copy.replace` (3.13) for IR-node rebuilding in optimizer passes,
  `warnings.deprecated` (3.13) for migration shims, t-strings / PEP 649
  lazy annotations (3.14).
- Decision rule: **no `sys.version_info` feature branches in library code**
  — a ~4,000-line library should not carry dual code paths across three
  supported versions. These go on a documented wishlist for a future 3.13
  floor. Everything adopted now is forward-compatible. Sole exception: the
  `ast.TemplateStr` grammar guard in §7.1.

### 7.4 Gotchas

1. **`IntEnum` vs `ast.unparse`**: `ast.unparse(ast.Constant(value=ArithOp.ADD))`
   emits `repr(ArithOp.ADD)` → `<ArithOp.ADD: 2>`, corrupting generated
   source. Convert to plain int at the `Literal` boundary
   (`Literal(int(ArithOp.ADD), "int")`). Zero change to generated code.
2. **Optional follow-up (separate commit, not part of no-drift phases)**:
   generated `_vec_arith(xp, 1, x, y)` is unreadable; a `StrEnum` with
   string ids (`_vec_arith(xp, "add", x, y)`) fits the "readable, inspectable
   output" goal — but it changes generated source, so it ships with
   `pytest --update-golden` and a deliberate golden regeneration.
3. Golden snapshots compare generated code via `ast.dump` equality and only
   assert on **generated** output — modernizing the compiler's own source
   (match, slots, type aliases) never touches them.
4. `filterwarnings = ["error", ...]` (upstream) means any accidental
   `DeprecationWarning` from new stdlib usage fails CI — an extra safety net
   for the modernization sweep.

## 8. Explicitly unchanged

- Public API surface: `vectorize`, `get_source`, `VectorizationError`,
  `python -m vectorizer.fuzz`, `py.typed`.
- Generated code output during the no-drift phases (asserted by golden
  snapshots + fuzzer + differential tests).
- All documented semantics, README tables, `docs/semantics.md`.
- Quality gates: ruff, mypy strict, 95% branch coverage, pytest
  warnings-as-errors, CI fuzz/docs jobs, `make bench` (imports updated only).
- `tests/golden/cases/` file format and `--update-golden` workflow.

## 9. Phased migration plan

Each phase ends with all gates green (`make check`, `make fuzz`, and on the
phase that moves runtime helpers also `make bench`).

| Phase | Content | Risk |
|---|---|---|
| **0** | Provision 3.12+ toolchain in dev env; fix the `docs/architecture.md` pipeline-order drift (§3.4); baseline green | — |
| **1** | **Floor bump + modernization sweep** (§7): pyproject/CI/README/docs checklist; ruff `--target-version py312` UP autofix; convert dispatch chains to `match` + `assert_never`; `type` aliases + `Literal` kinds; `slots=True`; `IntEnum` (with §7.4.1 int conversion); §7.1 validator coverage + version-parametrized rejection tests | Medium-low; golden snapshots must stay byte-stable |
| **2** | **Pure moves**: create `ir/`, `frontend/`, `optimize/`, `codegen/`; move modules; one-line re-export shims at old paths so tests keep importing; update docs/architecture.md in the same commit | Very low — mechanical |
| **3** | **Split `_runtime`** into `runtime/` (call-time) + `emit.py` (generation-time); introduce `runtime/registry.py`; switch lowering to registry keys instead of hardcoded `"_vec_minmax"` strings; update `test_bench.py` imports; golden snapshots must stay identical | Low |
| **4** | **Split `_lower`** into `lower/` submodules, initially keeping `_Lowerer` intact (methods move, state stays); introduce `kinds.py` with `VarInfo` and migrate the five dicts **one at a time** (`name_kinds` first, `_maybe_bool_names` last), full suite between each | Medium — protected by ~220 review regressions + differential + fuzz |
| **5** | **Dissolve tests** into the §6 layout: extend `support.py`/`conftest.py` (strict fixture, corpus loader); move round/milestone/coverage tests into themed files verbatim; remove phase-2 shims and fix remaining imports | Low |
| **6** | Move orchestration/helper-cache from `__init__` to `api.py`/`pipeline.py`; extract the `protect_domains` pass; final docs pass over `docs/architecture.md` + `docs/development.md` | Low |

Ordering rationale: modernization (phase 1) lands **before** the structural
moves so every file is written once in its final idiom; phases 3–4 then get
cheaper because `match`-based dispatch and `Literal`-typed kinds are exactly
the scaffolding `ir/walk.py` and `lower/kinds.py` need.

Optional follow-up commit after phase 6 (deliberate, visible change):
`StrEnum` string ids in generated code + golden regeneration (§7.4.2);
weakref-based helper cache; `TypeIs`/`copy.replace` when the floor moves to
3.13.

## 10. Risks and mitigations

- **Behavioral drift in phases 1 and 4** (modernization; kind inference):
  golden snapshots (ast.dump — formatting-tolerant but semantics-exact),
  ~220 review-round regressions, Hypothesis differential tests, and the
  fuzzer on fixed seeds form the safety net; migrate one tracking dict at a
  time in phase 4.
- **Import-path churn**: all moved modules are private; phase-2 re-export
  shims give a deprecation window, removed in phase 5. `test_bench.py` is
  the only internal consumer of private runtime members and is updated
  atomically.
- **Version-matrix gaps**: the new `ast.TemplateStr`/`type_params` rejection
  tests only exercise their real grammars on 3.14/3.12 respectively — the CI
  matrix (3.12/3.13/3.14) is what makes them meaningful; keep all three jobs.
- **Over-fragmentation**: `lower/` and `runtime/` are the only justified
  package splits (1,107 and 487 lines, distinct concerns); everything else
  stays single-file. Fallback if `lower/` proves awkward as a package:
  three flat modules (`lower.py`, `lower_kinds.py`, `lower_loops.py`) — the
  essential change is extracting the kind lattice, not the directory shape.
- **Docs drift** (upstream just demonstrated how easily it happens — §3.4):
  every module-moving phase updates `docs/architecture.md` in the same
  commit, and phase 6 ends with a full docs pass.
