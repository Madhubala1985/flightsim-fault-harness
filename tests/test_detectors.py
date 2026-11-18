"""Phase 3 tests: residual detectors stay quiet on a clean signal, stuck airspeed is caught."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fsfh.detectors import RESIDUAL_DETECTORS, DetectorModel, alarms_from_scores, residual_scores
from fsfh.evaluate import events_after
from fsfh.runlog import read_run

MODELS = Path(__file__).resolve().parents[1] / "models"
FIXTURES = Path(__file__).resolve().parents[1] / "runs" / "fixtures"
pytestmark = pytest.mark.skipif(not MODELS.is_dir(), reason="run `make fit` first")


def clean_cruise(duration_s: float = 300.0, seed: int = 0) -> pd.DataFrame:
    """Straight-and-level cruise with nominal sensor noise, consistent by construction."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * 10)
    t = np.arange(n) / 10.0
    return pd.DataFrame(
        {
            "t_s": t,
            "airspeed_kts": np.round((100 + rng.normal(0, 0.5, n)) / 0.1) * 0.1,
            "altitude_ft": np.round(6000 + rng.normal(0, 3.0, n)),
            "vertical_speed_fpm": np.round(rng.normal(0, 20.0, n) / 10) * 10,
            "pitch_deg": np.round(0.8 + rng.normal(0, 0.1, n), 2),
            "roll_deg": np.round(rng.normal(0, 0.1, n), 2),
            "heading_deg": np.mod(np.round(rng.normal(0, 0.3, n), 1), 360),
            "rpm": np.round((2450 + rng.normal(0, 5.0, n)) / 10) * 10,
        }
    )


def test_residual_detectors_never_fire_on_clean_signal() -> None:
    model = DetectorModel.load(MODELS)
    scores = residual_scores(clean_cruise(), model)
    scores["iforest"] = 0.0  # not under test here: it is data-driven, not a physics check
    alarms = alarms_from_scores(scores, model)
    for name in RESIDUAL_DETECTORS:
        assert not alarms[name].any(), f"{name} fired on a clean signal"


def test_stuck_airspeed_detected_within_10s() -> None:
    from fsfh.detectors import alarm_events, detect

    fixture = next(p for p in sorted(FIXTURES.iterdir()) if "stuck_airspeed" in p.name)
    run = read_run(fixture)
    _, alarms = detect(run.sensors, MODELS)
    fault = run.faults[0]
    events = alarm_events(alarms["any"].to_numpy(), alarms["t_s"].to_numpy())
    ttd = events_after(events, fault.onset_s, fault.end_s)
    assert ttd is not None and ttd <= 10.0, f"time to detect {ttd}"
