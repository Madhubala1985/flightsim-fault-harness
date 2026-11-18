PY ?= .venv/bin/python
TRIAGE_BACKEND ?= ollama
TRIAGE_MODEL ?= llama3.2:3b

.PHONY: install lint test fly campaign fit triage results clean

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check .

test:
	$(PY) -m pytest -q

fly:
	$(PY) -m fsfh fly --scenario basic --seed 1 --out runs/

campaign:
	$(PY) -m fsfh campaign --n 150 --seed 7 --out runs/

fit:
	$(PY) -m fsfh fit --manifest runs/campaign7-manifest.csv --models models/

triage:
	$(PY) -m fsfh triage --manifest runs/campaign7-manifest.csv --models models/ \
		--cache triage_cache/ --backend $(TRIAGE_BACKEND) --model $(TRIAGE_MODEL)

results:
	$(PY) -m fsfh results --runs runs/ --readme README.md --out results/ \
		--manifest runs/campaign7-manifest.csv --models models/ \
		--triage-backend $(TRIAGE_BACKEND) --triage-model $(TRIAGE_MODEL)

clean:
	rm -rf runs/raw .pytest_cache .ruff_cache
