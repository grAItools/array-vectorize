# Development

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/): dependencies
are locked in `uv.lock`, the development Python version is pinned in
`.python-version`, and every `make` target runs through `uv run`, so no
virtualenv needs to be activated.

```bash
make install      # uv sync (dev group) + pre-commit hooks
```

Optional dependency groups: `backends` (jax, CPU torch), `docs`
(zensical), `notebooks` (marimo). The `make` targets that need them
(`backends`, `docs`, `notebook`, `smoke`) select them automatically.
After editing dependencies in `pyproject.toml`, run `uv lock` and commit
`uv.lock`. The lock pins the newest compatible versions; `make lowest`
proves the declared minimums by running the suite with every direct
dependency at its floor (`--resolution lowest-direct`) on Python 3.12.

## Commands

```bash
make check        # ruff + mypy (strict) + pytest + coverage (gate: 95%)
make check PY=3.13    # any target on another Python (env: .venv-3.13)
make lint type test   # the same gates without coverage
make lowest       # the test suite at the minimum dependency versions
make fmt          # format + autofix
make smoke        # run scripts/smoke.py, examples/demo.py, the notebooks
make fuzz         # grammar fuzzer on the CI seeds
make bench        # pytest-benchmark suite (timings + dispatch gates)
make backends     # jax.jit / torch.compile compatibility tests
make docs         # build this site (zensical, --strict)
make docs-serve   # live-reload preview
make notebook     # open the marimo example notebooks
uv run pytest --update-golden    # regenerate golden source snapshots
uv run python -m array_vectorize.fuzz --seconds 60 --seed 0    # fuzz longer
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
  (Hypothesis derandomized), then `make smoke`.
- **lowest** — `make lowest`: the suite at the declared dependency floors.
- **fuzz** — the five fixed fuzzer seeds.
- **compile** — proves the generated source traces correctly under
  `jax.jit` and `torch.compile` (jit/compile compatibility of the
  generated code). Compiling detected-mode (unpinned) functions with
  `torch.compile` additionally requires `array-api-compat>=1.15`
  ("array_namespace can now be used under torch.compile"); pinned mode
  (`namespace=`) has no such requirement.
- **docs** — builds this site with `--strict`.
- **pages** — after CI passes on a push to `main`, deploys this site to
  [GitHub Pages](https://grAItools.github.io/array-vectorize/).

## Coding agents

Instructions and skills for coding agents are harness-agnostic; each
harness's own directory only forwards to them:

| Path | Role |
|---|---|
| `AGENTS.md` | the project instructions every agent reads |
| `.agents/skills/<name>/SKILL.md` | shared skills ([Agent Skills](https://agentskills.io) format, standard frontmatter only): `add-construct`, `semantics-review` |
| `CLAUDE.md` | Claude Code adapter: imports `AGENTS.md` |
| `.claude/skills/<name>` | Claude Code adapter: symlink to `.agents/skills/<name>` |
| `.claude/agents/semantics-reviewer.md` | Claude Code subagent that preloads the `semantics-review` skill |
| `.claude/settings.json`, `.claude/hooks/` | Claude Code only: command allowlist, a format-on-edit hook, and a stop hook that runs `make lint type` |

Edit the shared files; the adapters pick up changes through the import
and the symlinks. `tests/test_agent_layout.py` fails when a skill uses
harness-specific frontmatter or lacks its `.claude/skills` link. On
Windows, the symlinks need `git config core.symlinks true` (and Developer
Mode or an elevated shell) to check out as links.

## History

The original design document, the restructuring plan, and the log of
the 26-round adversarial review loop that drove the exactness work live
in git history; the design decisions that still hold are summarized in
[Architecture](architecture.md#design-decisions).
