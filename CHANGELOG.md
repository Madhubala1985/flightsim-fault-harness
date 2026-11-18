# Changelog

## 0.1.0 - Phase 1 (2026-11-18)

- `fsfh` package: JSBSim Cessna 172 (`c172x`) flown by a three-loop PID autopilot
  (altitude via elevator, heading via aileron, airspeed via throttle) around JSBSim's
  trimmed state.
- `basic` scenario: climb 5000 to 6000 ft, hold, turn to 090, turn back to 360,
  descend to 5000 ft.
- Truth logged at 10 Hz to Parquet with a JSON sidecar; noisy sensor channels derived
  from truth with a seeded RNG.
- CLI `python -m fsfh fly` and `python -m fsfh results`; `make results` regenerates the
  README table.
- Tests for property names, control sign conventions, altitude hold, heading capture,
  clean logs, reproducibility and the CLI smoke flight. GitHub Actions CI.

## 0.2.0 - Phase 2 (2026-11-18)

- `faults.py`: `Fault(type, onset_s, duration_s, magnitude)` and seven fault types.
  Sensor-level: stuck_airspeed, altimeter_bias, altimeter_drift, attitude_noise_burst,
  sensor_dropout. Aircraft-level, injected between autopilot and JSBSim: engine_power_loss
  (throttle authority scaled down) and control_lag (elevator command FIFO delay).
- Every run now writes `faults.json`, the ground-truth fault log.
- `python -m fsfh campaign --n 60 --seed 7`: mix of nominal and single-fault runs with
  randomised onset, duration and magnitude, plus `runs/campaign<seed>-manifest.csv`.
- `python -m fsfh plot <run_dir>`: truth vs sensor channels with fault windows shaded.
- Tests: stuck airspeed is flat while truth moves, dropout and drift arithmetic, engine
  power loss changes the flight, campaign manifest matches files on disk.

## 0.3.0 - Phase 3 (2026-11-18)

- `detectors.py`: four physics-consistency residual detectors on sensor channels only
  (altitude rate vs vertical speed, airspeed trend vs pitch/RPM, specific energy rate,
  staleness) plus an Isolation Forest on 5 s windowed features trained on nominal runs.
  Thresholds chosen on validation nominal runs for a target of 1 false alarm per hour.
- `evaluate.py`: per fault type detection rate and median time-to-detect, false alarms per
  flight hour on held-out nominal runs, detector x fault confusion table and figure.
- `python -m fsfh fit`; `make results` now writes `results/*.md`, `results/confusion.png`
  and splices both tables into the README.
- `runs/fixtures/`: five recorded faulted runs (sensors + fault log) with
  `expectations.json`; `tests/test_replay.py` is the replay-before-release gate.
- Tests: residual detectors never fire on a clean synthetic cruise; stuck airspeed is
  detected within 10 s on a fixture.

## 0.4.0 - Phase 4 (2026-11-18)

- `fsfh.triage`: LLM triage agent. Input is the alarm list of a flagged window plus a
  statistical summary of the sensor channels; output is a pydantic-validated
  `TriageReport` (fault_type or "unknown", confidence, evidence quotes, recommended check).
- Backends behind one interface: Anthropic (structured output via `messages.parse`, key
  from `.env`) and any OpenAI-compatible endpoint such as Ollama. Disk cache keyed on the
  full request; `--offline` serves cache only.
- `python -m fsfh triage` scores the agent on the campaign: accuracy per fault type,
  "unknown" rate on nominal windows, confidence calibration (ECE). `make results` adds the
  triage section from cached responses only, never the network.
- Tests: schema validation, cache round-trip with a fake backend, agent on a fixture from
  committed cached responses.

## 0.5.0 - Phase 5 (2026-11-18)

- README rewritten: one-paragraph summary, Mermaid architecture, three-command run, generated
  results only, "What the detectors cannot see", Limitations, "How to add a fault type".
- MIT LICENSE, CITATION.cff, CI badge. Per-run flight table moved out of the README to
  `results/runs.md`.

## 0.5.1 - Review fixes (2026-11-18)
- Bigger campaign (150 runs) and a pooled false-alarm estimate: held-out
  nominal hours plus pre-onset hours on faulted runs, with a per-fault
  "pre_onset_per_hour" column. The generated sentence replaces the
  misleading "0.00 per hour on 1.1 hours" reading (~1.1/hour pooled, the
  target rate).
- README: why the autopilot flies on truth (isolates sensor from aircraft
  faults for scoring; closing the loop is the next step).

## 0.5.2 - Triage via local model (2026-11-18)
- Triage table filled: 261 windows scored against ground truth with a free
  local model (ollama, llama3.2:3b), cache committed so `make results`
  reproduces the table offline.
- OllamaBackend uses constrained decoding (json_schema response format);
  free-form JSON mode let small models echo the schema instead of filling it.
- Makefile: TRIAGE_BACKEND / TRIAGE_MODEL variables (default ollama /
  llama3.2:3b); pass TRIAGE_BACKEND=anthropic for the Claude backend.
