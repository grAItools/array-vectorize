.PHONY: install fmt lint type test coverage check

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
