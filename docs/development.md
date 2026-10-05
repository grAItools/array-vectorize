# Development

## Commands

```bash
make check        # ruff + mypy (strict) + pytest + coverage (gate: 95%)
make fmt          # format + autofix
make fuzz         # grammar fuzzer on the CI seeds
make bench        # pytest-benchmark suite (timings + dispatch gates)
make docs         # build this site (zensical, --strict)
make docs-serve   # live-reload preview
make notebook     # open the marimo example notebooks
pytest --update-golden    # regenerate golden source snapshots
```

The default `make check` also runs the slow-marked performance gate;
plain `pytest` skips nothing but keeps benchmarks disabled
(`--benchmark-disable` is in `addopts`).

## Testing layers

| Layer | Where |
|---|---|
| unit (per source module: frontend, ir, lower, optimize, codegen, runtime) | `tests/unit/` |
| behavior (black-box semantics on NumPy) | `tests/behavior/` |
| features (public options: fallback, protect, verify) | `tests/features/` |
| backends (array-api-strict) | `tests/backends/` |
| differential (Hypothesis, edge values) | `tests/differential/` |
| golden source snapshots | `tests/golden/` (`cases/`, ast.dump equality) |
| inspectability (getsource, linecache) | `tests/test_inspectability.py` |
| grammar fuzzer | `src/array_vectorize/fuzz.py` + CI seeds |
| fuzz smoke | `tests/test_fuzz_smoke.py` |
| performance gate + benchmarks | `tests/test_perf.py`, `tests/test_bench.py` |

The historical review-round regression tests are dissolved into
`behavior/` and `features/` (provenance in git history).

Warnings are errors in the test suite, except numpy's data-dependent
numeric warnings (`overflow encountered`, `invalid value`,
`divide by zero`) which are expected on documented divergence paths.

## CI

- **check** — the full `make check` gate on Python 3.12–3.14
  (Hypothesis derandomized).
- **fuzz** — the five fixed fuzzer seeds.
- **compile** — proves the generated source traces correctly under
  `jax.jit` and `torch.compile` (jit/compile compatibility of the
  generated code). Compiling detected-mode (unpinned) functions with
  `torch.compile` additionally requires `array-api-compat>=1.15`
  ("array_namespace can now be used under torch.compile"); pinned mode
  (`namespace=`) has no such requirement.
- **docs** — builds this site with `--strict`.
- **pages** — runs the test suite and deploys this site to
  [GitHub Pages](https://grAItools.github.io/array-vectorize/) on every
  push to `main`.

## History

- `PLAN.md` — the original design document and milestone history.
- `REVIEW_OPEN.md` — the log of the 26-round adversarial review loop
  that drove the exactness work, with one section per round.
