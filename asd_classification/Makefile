# Makefile -- ASD classification pipeline (Phase 0 smoke)
#
# NOTE: GNU make is not installed on the target Windows machine. `make.cmd` / `make.ps1`
# are shims that dispatch to exactly the same CLI, so both entry points behave identically.

PY ?= D:/Anaconda/envs/paper/python.exe
export PYTHONPATH := src

.PHONY: all smoke validate test checklist intake align-check extract-real clean env

all: test smoke validate checklist

smoke:
	$(PY) -m asdpipe.cli.main smoke

validate:
	$(PY) -m asdpipe.cli.main validate

checklist:
	$(PY) -m asdpipe.cli.main checklist

intake:
	$(PY) -m asdpipe.cli.main intake

align-check:
	$(PY) -m asdpipe.cli.main align-check

extract-real:
	$(PY) -m asdpipe.cli.main extract-real

test:
	$(PY) -m pytest -q tests

env:
	$(PY) -m pip install -r requirements.txt

clean:
	$(PY) -m asdpipe.cli.main clean
