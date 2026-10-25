PY ?= python
export PYTHONPATH := src

.PHONY: all ingest run test
all: ingest run
ingest:
	$(PY) -m eardrift.ingest
run:
	$(PY) -m eardrift.pipeline
test:
	$(PY) -m pytest -q tests
