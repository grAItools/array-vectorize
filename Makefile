.PHONY: install fmt lint type test coverage check fuzz bench backends docs docs-serve notebook

install:
	python -m pip install -e ".[dev]"

fmt:
	python -m ruff check --fix src tests
	python -m ruff format src tests

lint:
	python -m ruff check src tests
	python -m ruff format --check src tests

type:
	python -m mypy

test:
	python -m pytest

coverage:
	python -m coverage run -m pytest
	python -m coverage report

check: lint type coverage

fuzz:
	for seed in 42 7 123 999 2024; do \
		python -m array_vectorize.fuzz --seed $$seed --cases 400 || exit 1; \
	done

bench:
	python -m pytest tests/test_bench.py -m slow --benchmark-enable \
		--benchmark-only --benchmark-columns=min,median,ops \
		--benchmark-group-by=func

backends:
	python -m pytest tests/backends/test_jax_torch_compilation.py -v -rs

docs:
	python -m zensical build --strict

docs-serve:
	python -m zensical serve

notebook:
	python -m marimo edit examples/notebooks
