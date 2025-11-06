.PHONY: lint typecheck test test-slow test-all docs

lint:
	python -m ruff check .
	python -m ruff format --check .

typecheck:
	python -m mypy src/flu_peak --strict

test:
	python -m pytest -m "not slow"

test-slow:
	python -m pytest -m "slow"

test-all:
	python -m pytest

docs:
	python -m sphinx -b html docs/source docs/build/html

all: lint test

# Full check including typecheck and docs (requires pandas-stubs, scipy-stubs)
full: lint typecheck test docs
