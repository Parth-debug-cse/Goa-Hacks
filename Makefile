.PHONY: setup test run verify probe deploy calibrate clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest

setup:
	uv venv --python 3.11 $(VENV)
	uv pip install --python $(PYTHON) -r requirements.txt

test:
	$(PYTHON) -m pytest tests/unit -v

test-live:
	$(PYTHON) -m pytest tests/live -v -m live

run:
	$(PYTHON) -m pom run $(PHOTO) --consent-confirmed

verify:
	$(PYTHON) -m pom verify $(RECEIPT)

verify-tamper:
	$(PYTHON) -m pom verify $(RECEIPT) --tamper-demo

probe:
	$(PYTHON) -m pom probe --engine $(ENGINE)

deploy:
	$(PYTHON) contracts/deploy.py

calibrate:
	$(PYTHON) -m pom calibrate --pairs-dir data/pairs

clean:
	rm -rf out/ .pytest_cache/ __pycache__/ pom/__pycache__/
