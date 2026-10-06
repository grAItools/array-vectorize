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

.PHONY: install fmt lint type test coverage check lowest smoke fuzz bench backends docs docs-serve notebook

install:
	$(UV) sync
	$(RUN) pre-commit install

fmt:
	$(RUN) ruff check --fix $(PY_PATHS)
	$(RUN) ruff format $(PY_PATHS)

lint:
	$(RUN) ruff check $(PY_PATHS)
	$(RUN) ruff format --check $(PY_PATHS)

type:
	$(RUN) mypy

test:
	$(RUN) pytest

coverage:
	$(RUN) coverage run -m pytest
	$(RUN) coverage report

check: lint type coverage

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
	$(RUN) pytest tests/test_bench.py -m slow --benchmark-enable \
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
