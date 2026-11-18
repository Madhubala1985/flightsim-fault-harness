PY ?= .venv/bin/python

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
	$(PY) -m fsfh campaign --n 60 --seed 7 --out runs/

fit:
	$(PY) -m fsfh fit --manifest runs/campaign7-manifest.csv --models models/

triage:
	$(PY) -m fsfh triage --manifest runs/campaign7-manifest.csv --models models/ --cache triage_cache/

results:
	$(PY) -m fsfh results --runs runs/ --readme README.md --out results/ \
		--manifest runs/campaign7-manifest.csv --models models/

clean:
	rm -rf runs/raw .pytest_cache .ruff_cache
