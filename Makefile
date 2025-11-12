.PHONY: lint test test-slow test-all docs

lint:
	python -m ruff check .
	python -m ruff format --check .

test:
	python -m pytest -m "not slow"

test-slow:
	python -m pytest -m "slow"

test-all:
	python -m pytest

docs:
	python -m sphinx -b html docs/source docs/build/html

all: lint test
