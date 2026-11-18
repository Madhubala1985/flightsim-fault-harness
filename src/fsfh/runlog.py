"""Write and read a run: truth.parquet, sensors.parquet and a meta.json sidecar.

Layout: ``<out>/raw/<run_id>/``. ``runs/raw/`` is git-ignored; only derived
summaries (see ``fsfh.results``) are committed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from fsfh import __version__
from fsfh.autopilot import AutopilotGains
from fsfh.faults import Fault
from fsfh.scenarios import Scenario
from fsfh.sensors import SensorSpec, specs_to_dicts
from fsfh.sim import LOG_HZ, TRUTH_COLUMNS, FlightResult

TRUTH_FILE = "truth.parquet"
SENSORS_FILE = "sensors.parquet"
META_FILE = "meta.json"
FAULTS_FILE = "faults.json"


@dataclass
class RunData:
    path: Path
    meta: dict
    truth: pd.DataFrame | None  # fixtures under runs/fixtures/ store sensors only
    sensors: pd.DataFrame
    faults: list[Fault]


def run_id(scenario: Scenario, seed: int) -> str:
    return f"{scenario.name}-seed{seed}-{int(scenario.duration_s)}s"


def write_run(
    out_dir: Path,
    scenario: Scenario,
    seed: int,
    result: FlightResult,
    sensors: pd.DataFrame,
    gains: AutopilotGains,
    sensor_specs: tuple[SensorSpec, ...],
    faults: list[Fault] | None = None,
    name: str | None = None,
) -> Path:
    faults = faults or []
    run_dir = Path(out_dir) / "raw" / (name or run_id(scenario, seed))
    run_dir.mkdir(parents=True, exist_ok=True)
    result.truth.to_parquet(run_dir / TRUTH_FILE, index=False)
    sensors.to_parquet(run_dir / SENSORS_FILE, index=False)
    meta = {
        "run_id": run_dir.name,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "fsfh_version": __version__,
        "jsbsim_version": result.jsbsim_version,
        "seed": seed,
        "scenario": scenario.to_dict(),
        "sim_dt_s": result.dt_s,
        "log_hz": LOG_HZ,
        "trim": asdict(result.trim),
        "autopilot_gains": asdict(gains),
        "sensor_specs": specs_to_dicts(sensor_specs),
        "truth_columns": TRUTH_COLUMNS,
        "sensor_columns": list(sensors.columns),
        "n_samples": int(len(result.truth)),
    }
    (run_dir / META_FILE).write_text(json.dumps(meta, indent=2) + "\n")
    # Ground-truth fault log: exactly what was injected, with its active window.
    fault_log = {"faults": [f.to_dict() for f in faults]}
    (run_dir / FAULTS_FILE).write_text(json.dumps(fault_log, indent=2) + "\n")
    return run_dir


def read_run(run_dir: Path) -> RunData:
    run_dir = Path(run_dir)
    return RunData(
        path=run_dir,
        meta=json.loads((run_dir / META_FILE).read_text()),
        truth=pd.read_parquet(run_dir / TRUTH_FILE) if (run_dir / TRUTH_FILE).is_file() else None,
        sensors=pd.read_parquet(run_dir / SENSORS_FILE),
        faults=read_faults(run_dir),
    )


def read_faults(run_dir: Path) -> list[Fault]:
    path = Path(run_dir) / FAULTS_FILE
    if not path.is_file():
        return []
    return [
        Fault(d["type"], d["onset_s"], d["duration_s"], d["magnitude"])
        for d in json.loads(path.read_text())["faults"]
    ]


def list_runs(out_dir: Path) -> list[Path]:
    raw = Path(out_dir) / "raw"
    if not raw.is_dir():
        return []
    return sorted(p for p in raw.iterdir() if (p / META_FILE).is_file())
