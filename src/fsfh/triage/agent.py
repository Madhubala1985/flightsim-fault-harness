"""Build the prompt for one flagged window and return a validated TriageReport."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fsfh.detectors import detect
from fsfh.faults import FAULT_TYPES, MAGNITUDE_UNITS
from fsfh.triage.backends import CachedBackend, validate
from fsfh.triage.features import alarm_summary, channel_summary, format_window
from fsfh.triage.schema import TriageReport

WINDOW_BEFORE_S = 15.0
WINDOW_AFTER_S = 30.0

SYSTEM_PROMPT = f"""You are a triage engineer for a flight-simulator fault-injection test bench.
The aircraft is a Cessna 172 flown by an autopilot through a scripted profile (climb, level turns,
descent) at about 100 kts. You receive, for one time window: the alarms raised by a set of fault
detectors, and a statistical summary of the sensor channels. You never see ground truth.

Detectors (scores are in nominal-sigma; an alarm means the score stayed above threshold):
- alt_rate: slope of the altitude channel disagrees with the vertical-speed channel.
- airspeed: airspeed trend disagrees with what pitch, RPM and vertical speed predict.
- energy: specific energy rate from the sensors disagrees with RPM and airspeed.
- staleness: some channel has shown the same value (or NaN) for more than 3 s.
- iforest: Isolation Forest on windowed channel statistics finds the window unusual.

Known fault types and the unit of their magnitude:
{chr(10).join(f"- {t}: {MAGNITUDE_UNITS[t]}" for t in FAULT_TYPES)}

Reasoning hints: a frozen channel with 1 distinct value and a staleness alarm points at a pitot
block (airspeed) or a dropout (all channels NaN). alt_rate alone, with vertical speed near zero but
altitude sloping, points at the altimeter (bias if it jumped once, drift if it keeps sloping).
airspeed and energy together with falling RPM point at engine power loss. airspeed, energy, alt_rate
and iforest together with oscillating pitch and vertical speed point at control lag. Attitude noise
shows as unusually high pitch/roll std with no residual alarms.

Answer 'unknown' when the evidence does not single out a fault, and keep confidence honest: it is
the probability that your fault_type is correct. Quote the specific alarms and channel statistics
you used in 'evidence'."""


def flagged_windows(alarms: pd.DataFrame, duration_s: float) -> list[tuple[float, float]]:
    """One window per rising edge of the combined alarm, clipped to the run."""
    from fsfh.detectors import alarm_events

    windows = []
    for t in alarm_events(alarms["any"].to_numpy(), alarms["t_s"].to_numpy()):
        t0, t1 = max(0.0, t - WINDOW_BEFORE_S), min(duration_s, t + WINDOW_AFTER_S)
        if not windows or t0 >= windows[-1][1]:
            windows.append((t0, t1))
    return windows


def describe_window(
    sensors: pd.DataFrame, scores: pd.DataFrame, alarms: pd.DataFrame, t0: float, t1: float
) -> str:
    return format_window(
        t0, t1, alarm_summary(scores, alarms, t0, t1), channel_summary(sensors, t0, t1)
    )


def triage_window(backend: CachedBackend, description: str) -> TriageReport:
    return validate(backend.complete(SYSTEM_PROMPT, description))


def triage_run(
    sensors: pd.DataFrame, model_dir: Path, backend: CachedBackend, windows=None
) -> list[tuple[float, float, TriageReport]]:
    """Triage every flagged window of one run (or the given windows)."""
    scores, alarms = detect(sensors, model_dir)
    duration = float(sensors.t_s.iloc[-1]) + 0.1
    windows = windows if windows is not None else flagged_windows(alarms, duration)
    out = []
    for t0, t1 in windows:
        report = triage_window(backend, describe_window(sensors, scores, alarms, t0, t1))
        out.append((t0, t1, report))
    return out
