"""Phase 2 tests: fault injection and campaign manifest."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from fsfh.campaign import run_campaign
from fsfh.faults import Fault, apply_sensor_faults
from fsfh.runlog import read_run
from fsfh.scenarios import BASIC
from fsfh.sensors import derive_sensors
from fsfh.sim import fly


@pytest.fixture(scope="module")
def short_truth():
    return fly(BASIC.truncated(60)).truth


def test_stuck_airspeed_flat_while_truth_changes(short_truth) -> None:
    fault = Fault("stuck_airspeed", onset_s=10.0, duration_s=30.0, magnitude=0.0)
    sensors = apply_sensor_faults(derive_sensors(short_truth, seed=3), [fault], seed=3)
    active = (sensors.t_s >= 10.0) & (sensors.t_s < 40.0)
    assert sensors.airspeed_kts[active].nunique() == 1
    assert short_truth.ias_kts[active].std() > 0.1  # truth is still moving (initial climb)
    assert sensors.airspeed_kts[~active].nunique() > 1  # outside the window it is normal


def test_dropout_and_drift(short_truth) -> None:
    faults = [
        Fault("sensor_dropout", 5.0, 2.0, 0.0),
        Fault("altimeter_drift", 20.0, 10.0, 2.0),
    ]
    clean = derive_sensors(short_truth, seed=3)
    sensors = apply_sensor_faults(clean, faults, seed=3)
    assert sensors[(sensors.t_s >= 5) & (sensors.t_s < 7)].drop(columns="t_s").isna().all().all()
    at_end = sensors.t_s.sub(29.9).abs().idxmin()
    assert np.isclose(sensors.altitude_ft[at_end] - clean.altitude_ft[at_end], 2.0 * 9.9)


def test_engine_power_loss_changes_flight() -> None:
    nominal = fly(BASIC.truncated(60)).truth
    faulted = fly(BASIC.truncated(60), faults=[Fault("engine_power_loss", 20.0, 40.0, 0.6)]).truth
    assert np.allclose(nominal.rpm[nominal.t_s < 20], faulted.rpm[faulted.t_s < 20])
    assert faulted.rpm[faulted.t_s > 30].mean() < nominal.rpm[nominal.t_s > 30].mean() - 100


def test_campaign_manifest_matches_disk(tmp_path: Path) -> None:
    manifest = run_campaign(BASIC.truncated(30), n=4, seed=7, out_dir=tmp_path)
    rows = list(csv.DictReader(manifest.open()))
    assert len(rows) == 4
    for row in rows:
        run = read_run(Path(row["path"]))
        assert run.meta["run_id"] == row["run_id"]
        if row["fault_type"] == "nominal":
            assert run.faults == []
        else:
            assert len(run.faults) == 1
            assert run.faults[0].type == row["fault_type"]
            assert run.faults[0].onset_s == float(row["onset_s"])
    assert {p.name for p in (tmp_path / "raw").iterdir()} == {r["run_id"] for r in rows}
