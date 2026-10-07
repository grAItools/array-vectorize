# Development

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/): dependencies
are locked in `uv.lock`, the development Python version is pinned in
`.python-version`, and every `make` target runs through `uv run`, so no
virtualenv needs to be activated.

```bash
make install      # uv sync (dev group) + prek hooks
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
make hooks        # all Git-tracked files through prek, with diffs on failure
make check        # cleanporter + ruff + four type checkers + pytest + coverage (95%)
make check PY=3.13    # any target on another Python (env: .venv-3.13)
make lint type test   # the same gates without coverage
make type         # mypy + pyright + zuban + pyrefly
make type-mypy    # individual checker (also: type-pyright, type-zuban, type-pyrefly)
make lowest       # the test suite at the minimum dependency versions
make fmt          # headers, cleanporter --fix, then ruff format / check --fix / format
make headers      # check canonical headers, including untracked Python files
make headers-fix  # apply canonical headers without changing source bodies
make smoke        # run scripts/smoke.py, examples/demo.py, the notebooks
make fuzz         # grammar fuzzer on the CI seeds
make bench        # pytest-benchmark suite (timings + dispatch gates)
make backends     # jax.jit / torch.compile compatibility tests
make docs         # build this site (zensical, --strict)
make docs-serve   # live-reload preview
make notebook     # open the marimo example notebooks
make release VERSION=0.1.0    # validate + gate + build + installed artifact checks
make package-smoke VERSION=0.1.0    # test wheel and sdist already in dist/
uv run pytest --update-golden    # regenerate golden source snapshots
uv run python -m array_vectorize.fuzz --seconds 60 --seed 0    # fuzz longer
```

The default `make check` also runs the slow-marked performance gate;
plain `pytest` skips nothing but keeps benchmarks disabled
(`--benchmark-disable` is in `addopts`).

`make lint` includes the cleanporter import gate, and `make fmt` runs the
same check with `--fix`: cleanporter rewrites first, so ruff formats the
already-wrapped lines (its module prefixes push lines past 100 columns).

## Type checking

All four checkers are blocking gates in `make type`, `make check`, git hooks,
CI, and releases. Make owns their commands; `pyproject.toml` owns their scopes
and settings. The recipes use `uv run --frozen`, so they run the locked versions
without changing the lock. Hooks invoke the same Make targets and check the
whole configured scope, even when only one file is staged. Python/stub files,
the Makefile, hook configuration, and dependency/typing configuration trigger them.

| Checker | Scope | Mode |
|---|---|---|
| mypy | `src` and `scripts` | strict |
| Pyright | `src` and `scripts` | standard |
| Zuban | `src`, `scripts`, and maintained tests | strict, with test annotation requirements relaxed |
| Pyrefly | `src`, `scripts`, and maintained tests | default diagnostics, including unannotated bodies |

All target Python 3.12. Zuban has its own configuration rather than inheriting
mypy's source-only scope. The wider pair resolves `support` and `corpus` from
the `tests` import root, like pytest. They exclude compiler input data
(`tests/corpus.py`), generated golden cases, and the optional JAX/PyTorch
compilation test module; the golden test harness and ordinary backend tests
remain checked. JAX/PyTorch compilation tests run in the dedicated CI job.

Pyrefly disables default hidden-path exclusions so checkouts inside `.paseo`
or another hidden worktree directory still check tests. Run it in project
mode: passing positional paths bypasses configured exclusions. Read checker
output after changing configuration; some tools only warn about unknown keys.

Pyright and Pyrefly receive uv's active interpreter, so `make type PY=3.13`
and custom `UV_PROJECT_ENVIRONMENT` values resolve the correct environment.
Pyright's PyPI wrapper downloads Node on its first run, which can take longer.

Fix diagnostics with accurate annotations and narrowing. The compiler attaches
metadata to functions dynamically: tests use `support.with_metadata` to describe
that boundary without changing the public API. `support.binding` checks IR
statement shapes before accessing binding-only fields. Deliberate invalid-input
tests may need a narrowly documented typing boundary; do not hide ordinary code
behind blanket suppressions.

## Testing layers

| Layer | Where |
|---|---|
| unit (per source module: frontend, ir, lower, optimize, codegen, runtime) | `tests/unit/` |
| behavior (black-box semantics on NumPy) | `tests/behavior/` |
| features (public options: fallback, protect, verify) | `tests/features/` |
| backends (array-api-strict) | `tests/backends/` |
| differential (Hypothesis, edge values) | `tests/differential/` |
| golden source snapshots | `tests/golden/` (`cases/`, ast.dump equality) |
| inspectability (getsource, linecache, memoization, docstrings) | `tests/behavior/test_inspectability.py` etc. |
| grammar fuzzer | `src/array_vectorize/fuzz.py` + CI seeds |
| fuzz smoke | `tests/unit/test_fuzz.py` |
| performance gate + benchmarks | `tests/perf/` |

The historical review-round regression tests are dissolved into
`behavior/` and `features/` (provenance in git history).

Warnings are errors in the test suite, except numpy's data-dependent
numeric warnings (`overflow encountered`, `invalid value`,
`divide by zero`) which are expected on documented divergence paths.

## Style

Repository-owned Python and stub files carry the canonical template in
`.license-header.txt`, followed by one blank line. This includes golden snapshots,
marimo notebooks, tests, and hidden scripts. The notice identifies grAItools,
2026, and BSD-3-Clause; its year is updated centrally rather than per file.
`make headers` checks tracked and nonignored untracked files without writing;
`make headers-fix` applies the template, preserving shebangs, encoding declarations,
line endings, and source bodies. Conflicting ownership or license notices and source
symlinks are refused. Ignored environments, downloaded code, and embedded snippets
are outside the policy. The whole-tree hook runs even on template-only commits.
Golden snapshot writers preserve the header; generated function source is unchanged.

The code follows the [Google Python Style Guide]
(https://google.github.io/styleguide/pyguide.html). Ruff enforces it in
`make lint`: pydocstyle with the google convention, pep8-naming, import
conventions (`numpy` as `np`), absolute imports only (TID252,
`ban-relative-imports = "all"`), and additionally BLE001, PLW0603, G004,
PGH003 and PGH004 (evaluated: zero violations).

Imports follow Google §2.2 (import modules, not names), enforced by
[cleanporter](https://github.com/grAItools/cleanporter) in `make lint` and
`make fmt` (scope `all`; the golden cases and the notebooks are excluded
as generated and marimo-owned, `tests/corpus.py` is skipped as golden
test data, and `treat_unresolved_as_error` is on, so an import
cleanporter cannot classify fails the gate). The deliberate exceptions
are inline `# cleanporter: ignore[...]` suppressions, each naming its
finding code and carrying its reason in the suppression comment or the
comment block above it. One import
per line per §3.13 (ruff isort `force-single-line`).

Evaluated and deliberately skipped: TD (conflicts with Google's current
TODO format) and PLC0415 (import-inside-function is a deliberate pattern
in this compiler's test subjects).

The only deviation from the guide is the line length: 100 columns (the
guide says 80). Tests relax D1 (missing docstring) — test names are
self-documenting; `tests/corpus.py` ignores all D rules (golden-pinned
test data), and `tests/behavior/test_docstrings.py` ignores D4 (it holds
the NumPy-style docstring subject).

## CI

- **static** — `make lint type` once on Python 3.12, including all four checkers.
- **check** — `make coverage` on Python 3.12–3.14
  (Hypothesis derandomized), then `make smoke`.
- **windows** — correctness tests on Python 3.12 and 3.14 with locked dependencies
  and the deterministic Hypothesis profile. Git symlink support is configured before
  checkout so agent-layout checks remain enabled. Timing-sensitive performance tests
  stay on Linux; optional JAX/PyTorch compilation stays in its dedicated job.
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
- **release** — on a pushed `v*` tag, validates the version, lock, changelog,
  clean checkout and ancestry on `origin/main`, then runs the full gate. It builds
  once with `uv build --no-sources` and tests the wheel and sdist installed into
  independent temporary environments. A separate publishing job creates the GitHub
  release from those exact artifacts, with generated notes and prerelease marking.
  Build permissions are read-only; only publishing can write repository contents.
  Runs for the same tag queue rather than cancel during publishing.

## Releases

Versioning follows [Semantic Versioning](https://semver.org/). The version
lives in two places that must agree — `pyproject.toml` and
`src/array_vectorize/__init__.py` (`__version__`) — and is recorded in
`uv.lock`, so run `uv lock` after bumping. Every release gets an entry in
`CHANGELOG.md` (Keep a Changelog format, linked from the README and
`pyproject.toml`).

```bash
# 1. bump the version, update CHANGELOG.md, commit
# 2. verify, run the full gate, build dist/
make release VERSION=0.1.0
# 3. tag and push (the tag triggers the release workflow)
git tag -a v0.1.0 -m "v0.1.0"
git push origin main --tags
```

`make release` and the workflow share a stdlib validator. The tag must be `v`
plus the exact normalized public Python package version in `pyproject.toml`,
`__version__`, and the project entry in `uv.lock`; local version suffixes and
noncanonical spellings are refused. The changelog must contain that version.
The working tree must be clean, the lock fresh, and HEAD merged into `origin/main`
(fetch origin before releasing). Cleanliness is checked again immediately before
building. Keep `dist/` empty before building: artifact checks require exactly one
wheel and one sdist, preventing upload of leftovers from an earlier release.
Each artifact is installed separately outside the checkout; imports must come
from that environment, match the requested version, carry `py.typed`, and compile
and run a source-backed function correctly. Tagging remains manual; PyPI publishing
and workflow dispatch are not configured.

## Commit messages

Commit subjects follow [Conventional Commits 1.0.0]
(https://conventionalcommits.org): `type(scope): summary`, lowercase.
Types: `build`, `chore`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`,
`revert`, `style`, `test`. The scope is optional but recommended, from
the package areas: `codegen`, `lower`, `optimize`, `frontend`, `ir`,
`runtime`, `api`, `pipeline`, `fuzz`, `agents`. Mark breaking changes
with `!` after the type/scope, or with a `BREAKING CHANGE` footer. A
`commit-msg` hook (`conventional-pre-commit`) enforces the format.

The history before the switch used `area: summary` subjects; map them
to the new format like this:

| Old subject | Conventional equivalent |
|---|---|
| `codegen: x` | `feat(codegen): x` or `fix(codegen): x` |
| `agents: x` | `chore(agents): x` |
| `tests: x` | `test: x` (type, no scope) |
| `docs: x` | `docs: x` (type, no scope) |
| `ci: x` | `ci: x` (type, no scope) |
| `build: x` | `build: x` (type, no scope) |

The hook runner is `prek`; the configuration filename and Git's `pre-commit`
and `commit-msg` stage names stay unchanged. Existing clones must rerun
`make install` after switching runners to replace both installed hooks.
`make hooks` checks all Git-tracked files; staged-only runs and untracked new
files can be silently skipped by filename-filtered hooks. `make lint type`
walks maintained code directly, and the header hook additionally discovers
nonignored untracked Python files. CI uses the same Make checks without autofixing.

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
Windows, the symlinks need `git config core.symlinks true` before checkout (and Developer
Mode or an elevated shell) to check out as links.

## History

The original design document, the restructuring plan, and the log of
the 26-round adversarial review loop that drove the exactness work live
in git history; the design decisions that still hold are summarized in
[Architecture](architecture.md#design-decisions).

## Portable source files

Git checkouts use LF through `.gitattributes`. Repository text I/O names UTF-8
explicitly; generated snippet and snapshot writers name LF explicitly. Tests also
write CRLF and Unicode paths deliberately to verify source inspection on Windows.
No Make installation is required for Windows correctness testing: use
`uv run --frozen pytest -m "not slow" -ra` with `CI=1`.
