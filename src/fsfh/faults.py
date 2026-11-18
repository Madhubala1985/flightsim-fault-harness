"""Fault definitions and injection.

Two kinds of fault, distinguished by ``FAULT_TARGET``:

* **sensor** faults corrupt the sensor frame after the flight. Truth is
  untouched, so the ground-truth log stays exact.
* **aircraft** faults act inside the simulation loop (``fsfh.sim.fly``), so the
  flight itself changes and the autopilot has to cope.

``magnitude`` means something different per type; see ``MAGNITUDE_UNITS``.
A fault is active for ``onset_s <= t < onset_s + duration_s``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

FAULT_TARGET: dict[str, str] = {
    "stuck_airspeed": "sensor",  # pitot blocked: airspeed frozen at its onset value
    "altimeter_bias": "sensor",  # step offset on altitude
    "altimeter_drift": "sensor",  # ramp on altitude, growing from zero at onset
    "attitude_noise_burst": "sensor",  # extra white noise on pitch and roll
    "sensor_dropout": "sensor",  # all sensor channels NaN
    "engine_power_loss": "aircraft",  # throttle authority scaled down
    "control_lag": "aircraft",  # elevator command delayed
}

MAGNITUDE_UNITS: dict[str, str] = {
    "stuck_airspeed": "unused",
    "altimeter_bias": "ft",
    "altimeter_drift": "ft/s",
    "attitude_noise_burst": "deg (noise sd)",
    "sensor_dropout": "unused",
    "engine_power_loss": "fraction of throttle authority removed (0-1)",
    "control_lag": "s",
}

FAULT_TYPES: tuple[str, ...] = tuple(FAULT_TARGET)


@dataclass(frozen=True)
class Fault:
    type: str
    onset_s: float
    duration_s: float
    magnitude: float

    def __post_init__(self) -> None:
        if self.type not in FAULT_TARGET:
            raise ValueError(f"unknown fault type {self.type!r}; known: {FAULT_TYPES}")

    @property
    def end_s(self) -> float:
        return self.onset_s + self.duration_s

    @property
    def target(self) -> str:
        return FAULT_TARGET[self.type]

    def active(self, t: float) -> bool:
        return self.onset_s <= t < self.end_s

    def to_dict(self) -> dict:
        return {**asdict(self), "end_s": self.end_s, "target": self.target}


def sensor_faults(faults: list[Fault]) -> list[Fault]:
    return [f for f in faults if f.target == "sensor"]


def aircraft_faults(faults: list[Fault]) -> list[Fault]:
    return [f for f in faults if f.target == "aircraft"]


# --- sensor-level injection -------------------------------------------------


def apply_sensor_faults(sensors: pd.DataFrame, faults: list[Fault], seed: int) -> pd.DataFrame:
    """Return a corrupted copy of the sensor frame. Order of faults is preserved."""
    out = sensors.copy()
    t = out["t_s"].to_numpy()
    rng = np.random.default_rng(seed)
    for f in sensor_faults(faults):
        m = (t >= f.onset_s) & (t < f.end_s)
        if not m.any():
            continue
        if f.type == "stuck_airspeed":
            onset_idx = int(np.flatnonzero(m)[0])
            out.loc[m, "airspeed_kts"] = out["airspeed_kts"].iloc[onset_idx]
        elif f.type == "altimeter_bias":
            out.loc[m, "altitude_ft"] += f.magnitude
        elif f.type == "altimeter_drift":
            out.loc[m, "altitude_ft"] += f.magnitude * (t[m] - f.onset_s)
        elif f.type == "attitude_noise_burst":
            n = int(m.sum())
            out.loc[m, "pitch_deg"] += rng.normal(0.0, f.magnitude, n)
            out.loc[m, "roll_deg"] += rng.normal(0.0, f.magnitude, n)
        elif f.type == "sensor_dropout":
            channels = [c for c in out.columns if c != "t_s"]
            out.loc[m, channels] = np.nan
    return out


# --- aircraft-level injection ------------------------------------------------


class AircraftFaultInjector:
    """Modifies control commands on their way from the autopilot to JSBSim.

    * ``engine_power_loss``: the throttle reaching the engine is
      ``cmd * (1 - magnitude)``. This models a loss of available power by
      limiting throttle authority rather than editing a JSBSim engine property,
      because JSBSim's piston engine exposes no "partial power" knob in its
      property tree; only running/not-running and the throttle input.
    * ``control_lag``: the elevator command is delayed by ``magnitude`` seconds
      through a FIFO of ``round(magnitude / dt)`` steps.
    """

    def __init__(self, faults: list[Fault], dt: float) -> None:
        self.faults = aircraft_faults(faults)
        self.dt = dt
        self._elevator_fifo: deque[float] = deque()

    def throttle(self, t: float, cmd: float) -> float:
        for f in self.faults:
            if f.type == "engine_power_loss" and f.active(t):
                cmd *= 1.0 - f.magnitude
        return cmd

    def elevator(self, t: float, cmd: float) -> float:
        for f in self.faults:
            if f.type == "control_lag" and f.active(t):
                n = max(1, round(f.magnitude / self.dt))
                self._elevator_fifo.append(cmd)
                while len(self._elevator_fifo) > n:
                    self._elevator_fifo.popleft()
                return self._elevator_fifo[0]
        self._elevator_fifo.clear()
        return cmd
