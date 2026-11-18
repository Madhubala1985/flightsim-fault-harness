"""Generate a campaign: a mix of nominal and single-fault runs plus manifest.csv.

Every random choice comes from one ``default_rng(seed)``: whether a run is
faulted, which fault, its onset, duration and magnitude, and the per-run
sensor-noise seed. The same campaign seed therefore reproduces the same set of
runs. Onset is kept away from the first minute (the initial climb) and the end
of the flight so every fault has a nominal lead-in and an active window.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from fsfh.autopilot import AutopilotGains
from fsfh.faults import FAULT_TYPES, Fault, apply_sensor_faults
from fsfh.runlog import write_run
from fsfh.scenarios import Scenario
from fsfh.sensors import DEFAULT_SENSORS, derive_sensors
from fsfh.sim import fly

NOMINAL_FRACTION = 1 / 3
MANIFEST_COLUMNS = [
    "run_id",
    "path",
    "sensor_seed",
    "fault_type",
    "target",
    "onset_s",
    "duration_s",
    "magnitude",
]

# (low, high) magnitude ranges per fault type; units in fsfh.faults.MAGNITUDE_UNITS.
MAGNITUDE_RANGES: dict[str, tuple[float, float]] = {
    "stuck_airspeed": (0.0, 0.0),
    "altimeter_bias": (50.0, 400.0),
    "altimeter_drift": (0.5, 5.0),
    "attitude_noise_burst": (1.0, 5.0),
    "sensor_dropout": (0.0, 0.0),
    "engine_power_loss": (0.3, 0.8),
    "control_lag": (0.5, 2.0),
}


def draw_fault(rng: np.random.Generator, duration_s: float) -> Fault | None:
    if rng.random() < NOMINAL_FRACTION:
        return None
    ftype = str(rng.choice(FAULT_TYPES))
    lead_in = min(60.0, 0.25 * duration_s)
    onset = float(rng.uniform(lead_in, max(lead_in, duration_s - 2 * lead_in)))
    fault_len = float(rng.uniform(0.25 * lead_in, min(120.0, duration_s - onset)))
    lo, hi = MAGNITUDE_RANGES[ftype]
    return Fault(ftype, round(onset, 1), round(fault_len, 1), round(float(rng.uniform(lo, hi)), 3))


def run_campaign(scenario: Scenario, n: int, seed: int, out_dir: Path) -> Path:
    """Fly ``n`` runs and return the manifest path."""
    rng = np.random.default_rng(seed)
    gains = AutopilotGains()
    rows = []
    for i in range(n):
        fault = draw_fault(rng, scenario.duration_s)
        faults = [fault] if fault else []
        sensor_seed = int(rng.integers(0, 2**31 - 1))
        name = f"campaign{seed}-run{i:03d}"
        result = fly(scenario, gains, faults)
        sensors = derive_sensors(result.truth, sensor_seed, DEFAULT_SENSORS)
        sensors = apply_sensor_faults(sensors, faults, sensor_seed)
        run_dir = write_run(
            out_dir, scenario, sensor_seed, result, sensors, gains, DEFAULT_SENSORS, faults, name
        )
        rows.append(
            {
                "run_id": name,
                "path": str(run_dir),
                "sensor_seed": sensor_seed,
                "fault_type": fault.type if fault else "nominal",
                "target": fault.target if fault else "",
                "onset_s": fault.onset_s if fault else "",
                "duration_s": fault.duration_s if fault else "",
                "magnitude": fault.magnitude if fault else "",
            }
        )
        print(f"[{i + 1}/{n}] {name}: {rows[-1]['fault_type']}")
    manifest = Path(out_dir) / f"campaign{seed}-manifest.csv"
    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest
