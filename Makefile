# Every target runs through `uv run`, which uses the project's locked
# environment (.venv, Python from .python-version), so no virtualenv needs
# to be activated first.
UV ?= uv
RUN := $(UV) run

# `make check PY=3.13` runs any target on another Python, in its own
# environment (.venv-3.13), the way the CI matrix does
ifdef PY
export UV_PYTHON := $(PY)
export UV_PROJECT_ENVIRONMENT := .venv-$(PY)
endif
PY_PATHS := src tests scripts examples .claude/hooks

.PHONY: install fmt lint type type-mypy type-pyright type-zuban type-pyrefly test coverage check lowest smoke fuzz bench backends docs docs-serve notebook release

install:
	$(UV) sync
	$(RUN) pre-commit install

# fixers run to completion whatever findings remain; `make lint` stays
# the gate. cleanporter gets `test $$? -le 1`: it exits 1 whenever
# anything remains it will not rewrite (no exit-zero flag), but exit 2
# is an operational error and must fail the recipe. ruff check --fix
# gets --exit-zero: ruff cannot fix E501 itself, and cleanporter's
# module prefixes push lines past 100 columns; ruff format runs first
# and last so the fixes land on wrapped code
fmt:
	$(RUN) cleanporter --fix $(PY_PATHS); test $$? -le 1
	$(RUN) ruff format $(PY_PATHS)
	$(RUN) ruff check --fix --exit-zero $(PY_PATHS)
	$(RUN) ruff format $(PY_PATHS)

lint:
	$(RUN) ruff check $(PY_PATHS)
	$(RUN) ruff format --check $(PY_PATHS)
	# the import gate: exits 0 only when clean -- treat_unresolved_as_error
	# makes unclassifiable imports fail too
	$(RUN) cleanporter $(PY_PATHS)

type: type-mypy type-pyright type-zuban type-pyrefly

type-mypy:
	$(RUN) --frozen mypy

# Query uv's interpreter so PY= and UV_PROJECT_ENVIRONMENT work too.
type-pyright:
	$(RUN) --frozen pyright --pythonpath "$$($(RUN) --frozen python -c 'import sys; print(sys.executable)')"

type-zuban:
	$(RUN) --frozen zuban mypy

type-pyrefly:
	$(RUN) --frozen pyrefly check --python-interpreter-path "$$($(RUN) --frozen python -c 'import sys; print(sys.executable)')"

test:
	$(RUN) pytest

coverage:
	$(RUN) coverage run -m pytest
	$(RUN) coverage report

check: lint type coverage

# cut a release: the version must already be bumped (pyproject.toml,
# src/array_vectorize/__init__.py, uv.lock); runs the full gate and
# builds sdist + wheel into dist/. Tagging stays a manual step:
# git tag -a v$(VERSION) -m "v$(VERSION)"
release:
	@test "$(VERSION)" != "" || { echo "usage: make release VERSION=0.1.0" >&2; exit 1; }
	@grep -qx 'version = "$(VERSION)"' pyproject.toml || { echo "pyproject.toml is not at version $(VERSION)" >&2; exit 1; }
	@grep -qx '__version__ = "$(VERSION)"' src/array_vectorize/__init__.py || { echo "src/array_vectorize/__init__.py is not at version $(VERSION)" >&2; exit 1; }
	$(RUN) uv lock --check
	$(MAKE) check
	$(RUN) uv build

# the test suite with every direct dependency at the minimum pyproject.toml
# allows, on the minimum Python; proves the declared floors
lowest:
	rm -rf .venv-lowest
	$(UV) venv -q --python 3.12 .venv-lowest
	$(UV) pip install -q --python .venv-lowest --resolution lowest-direct -e . --group dev
	.venv-lowest/bin/python -m pytest

# the runnable examples are not part of the test suite; keep them working
smoke:
	$(RUN) python scripts/smoke.py
	$(RUN) python examples/demo.py
	for nb in examples/notebooks/*.py; do \
		$(RUN) --group notebooks python $$nb > /dev/null || exit 1; \
	done

fuzz:
	for seed in 42 7 123 999 2024; do \
		$(RUN) python -m array_vectorize.fuzz --seed $$seed --cases 400 || exit 1; \
	done

bench:
	$(RUN) pytest tests/perf/test_bench.py -m slow --benchmark-enable \
		--benchmark-only --benchmark-columns=min,median,ops \
		--benchmark-group-by=func

backends:
	$(RUN) --group backends pytest tests/backends/test_jax_torch_compilation.py -v -rs

docs:
	$(RUN) --group docs zensical build --strict

docs-serve:
	$(RUN) --group docs zensical serve

notebook:
	$(RUN) --group notebooks marimo edit examples/notebooks
