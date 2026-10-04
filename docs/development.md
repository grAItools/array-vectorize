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
| unit (IR, optimizer, codegen, validator) | `tests/test_*.py` |
| golden source snapshots | `tests/golden/cases/` (ast.dump equality) |
| differential (Hypothesis, edge values) | `tests/test_differential.py` |
| grammar fuzzer | `src/vectorizer/fuzz.py` + CI seeds |
| adversarial-review regressions | `tests/test_review_round1..25.py` |
| performance gate + benchmarks | `tests/test_perf.py`, `tests/test_bench.py` |

Warnings are errors in the test suite, except numpy's data-dependent
numeric warnings (`overflow encountered`, `invalid value`,
`divide by zero`) which are expected on documented divergence paths.

## CI

- **check** — the full `make check` gate on Python 3.12–3.14
  (Hypothesis derandomized).
- **fuzz** — the five fixed fuzzer seeds.
- **docs** — builds this site with `--strict`.

## History

- `PLAN.md` — the original design document and milestone history.
- `REVIEW_OPEN.md` — the log of the 26-round adversarial review loop
  that drove the exactness work, with one section per round.
