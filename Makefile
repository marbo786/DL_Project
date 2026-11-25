PYTHON ?= python
REPRODUCE_ARGS ?=

.PHONY: setup test lint demo reproduce

setup:
	$(PYTHON) scripts/setup.py

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check api research scripts tests web/frontend/scripts

demo:
	npm --prefix web/frontend run demo

reproduce:
	$(PYTHON) scripts/reproduce.py $(REPRODUCE_ARGS)
