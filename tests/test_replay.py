"""Replay-before-release gate.

Five recorded faulted runs live under runs/fixtures/ (sensors + fault log only).
runs/fixtures/expectations.json says, per run, the latest acceptable time-to-detect
and how many alarm events before onset are tolerated. Any change to the detectors
or the models must keep these passing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fsfh.detectors import alarm_events, detect
from fsfh.evaluate import events_after
from fsfh.runlog import read_run

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "runs" / "fixtures"
MODELS = ROOT / "models"
EXPECTATIONS = json.loads((FIXTURES / "expectations.json").read_text())


@pytest.mark.skipif(not MODELS.is_dir(), reason="run `make fit` first")
@pytest.mark.parametrize("run_name", sorted(EXPECTATIONS))
def test_replay_detects_recorded_fault(run_name: str) -> None:
    exp = EXPECTATIONS[run_name]
    run = read_run(FIXTURES / run_name)
    fault = run.faults[0]
    assert fault.type == exp["fault_type"]
    _, alarms = detect(run.sensors, MODELS)
    t = alarms["t_s"].to_numpy()
    events = alarm_events(alarms["any"].to_numpy(), t)
    ttd = events_after(events, fault.onset_s, fault.end_s)
    assert ttd is not None, f"{run_name}: {fault.type} not detected"
    assert ttd <= exp["max_ttd_s"], f"{run_name}: ttd {ttd:.1f}s > {exp['max_ttd_s']}s"
    pre_onset = sum(e < fault.onset_s for e in events)
    assert pre_onset <= exp["max_pre_onset_alarms"], f"{run_name}: {pre_onset} alarms before onset"
