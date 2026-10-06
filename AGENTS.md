# AGENTS.md

`array-vectorize` is a source-to-source compiler: `vectorize(f)` turns a
scalar Python function into an Array API function (NumPy, PyTorch, JAX,
CuPy, `array-api-strict`). Its contract is **exactness**: generated code
matches scalar Python on every lane, except for the divergences listed in
`docs/semantics.md`. Anything it cannot translate exactly, it rejects.

## Commands

Run everything through `make` or `uv run`. The system `python` may be
older than the required 3.12; uv supplies the pinned interpreter and the
locked environment (`uv.lock`).

- `make check`: the gate (ruff, strict mypy, pytest, 95% branch coverage),
  about 10 s. The work is done when it passes.
- `make lint type test`: the same gate without coverage, for the inner loop.
- `make fmt`: format and autofix.
- Area-specific: `make fuzz` (lowering, optimizer, runtime), `make backends`
  (codegen, namespace handling), `make docs` (anything in `docs/`),
  `make smoke` (public API, examples, notebooks).
- Other Python versions: `make check PY=3.13` (own `.venv-3.13`).
- Dependencies: edit `pyproject.toml`, run `uv lock`, commit `uv.lock`.
  Floors are real: when one changes, or a test needs a newer library,
  run `make lowest` and raise the floor until it passes.

## Before changing `src/`

Read `docs/architecture.md`. It covers the pipeline stages, the runtime
helpers, and design decisions D1–D10. Comments cite those decisions as
`design Dn`.

Invariants:

- **Reject over miscompile (D10).** When a construct cannot be lowered
  exactly, `frontend/validate.py` rejects it with a positioned diagnostic.
- **Backend neutrality (D9).** Generated code calls only `xp.*` and the
  runtime helpers registered in `runtime/registry.py`.
- **Layering**, enforced by `tests/unit/test_layering.py`: `runtime/` is
  a stdlib-only leaf, `ir/` depends on nothing in the package, only
  `__init__.py` imports `api`, and the only runtime dependency is
  `array-api-compat`.
- **Golden snapshots** (`tests/golden/cases/`) pin the generated source
  through `ast.dump` equality. Regenerate with
  `uv run pytest --update-golden` only when the generated code is meant to
  change, then review `git diff tests/golden/cases` line by line.

## Tests

The suite is organized by level and topic. Put each test in the file for
the behavior it exercises; regression tests read as behavior tests (name
the behavior, not the bug that found it).

| Testing | Goes in |
|---|---|
| one source module, white-box | `tests/unit/test_<module>.py` |
| end-to-end semantics on NumPy | `tests/behavior/test_<topic>.py` |
| a public option (`fallback`, `protect_domains`, `verify`, `namespace`) | `tests/features/` |
| `array-api-strict`, jax, torch | `tests/backends/` |
| benchmarks and the perf gate | `tests/perf/` |
| a new corpus function: its generated source | `tests/corpus.py` + `GOLDEN_NAMES`, then `--update-golden` |
| a new corpus function: values against scalar Python | `tests/differential/` (Hypothesis) |

- `tests/` is on `pythonpath`: import helpers as `from support import
  make_fn`, not `tests.support`.
- `vectorize` reads source with `inspect.getsource`, so test subjects must
  live in real files. Use `support.make_fn` / `make_module` for snippets.
- Warnings are errors (`filterwarnings = ["error", ...]` in `pyproject.toml`).

## Skills

Skills live in `.agents/skills/` in the [Agent Skills](https://agentskills.io)
format. Load the matching one before starting:

- Extending the supported subset: `.agents/skills/add-construct/SKILL.md`.
- Reviewing compiler changes for miscompiles:
  `.agents/skills/semantics-review/SKILL.md`.

## Agent configuration

The shared sources are `AGENTS.md` and `.agents/`. Harness directories only
adapt them: `CLAUDE.md` imports this file, `.claude/skills/<name>` symlinks
to `.agents/skills/<name>`, and `.claude/agents/` wraps skills as Claude
subagents. `.claude/settings.json` and `.claude/hooks/` hold Claude-only
permissions and hooks, which call the `make` targets above.
`tests/test_agent_layout.py` enforces this layout.

To add a skill, create `.agents/skills/<name>/SKILL.md` with only the
standard frontmatter fields (`name` matching the directory, `description`),
then link it for Claude:
`ln -s ../../.agents/skills/<name> .claude/skills/<name>`.

## Docs and commits

- `docs/` is the single source for the supported subset, semantics,
  divergences, and API. `README.md` only links to it. A user-visible
  change updates the matching page in the same commit; moving modules
  updates `docs/architecture.md`.
- Commit subjects follow [Conventional Commits 1.0.0]
  (https://conventionalcommits.org): `type(scope): summary`, lowercase.
  Types: `build`, `chore`, `ci`, `docs`, `feat`, `fix`, `perf`,
  `refactor`, `revert`, `style`, `test`. The scope is optional but
  recommended, from the package areas: `codegen`, `lower`, `optimize`,
  `frontend`, `ir`, `runtime`, `api`, `pipeline`, `fuzz`, `agents`.
  Mark breaking changes with `!` after the type/scope, or with a
  `BREAKING CHANGE` footer.
