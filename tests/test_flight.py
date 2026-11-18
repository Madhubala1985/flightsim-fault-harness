"""Phase 1 tests: property names, control signs, altitude hold, clean logs, reproducibility."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fsfh.autopilot import PID, wrap_deg
from fsfh.runlog import read_run
from fsfh.scenarios import BASIC
from fsfh.sensors import DEFAULT_SENSORS, derive_sensors
from fsfh.sim import PROPERTIES, fly, make_fdm

ALT_TOLERANCE_FT = 50.0
HDG_TOLERANCE_DEG = 2.0


@pytest.fixture(scope="module")
def basic_flight() -> pd.DataFrame:
    """One full ``basic`` flight shared by the tests in this module (~2 s wall time)."""
    return fly(BASIC).truth


def test_property_names_exist() -> None:
    fdm, _ = make_fdm(BASIC)
    for name in PROPERTIES.values():
        listed = {line.split(" ")[0] for line in fdm.query_property_catalog(name).splitlines()}
        assert name in listed, f"{name} not in JSBSim property catalog"


def test_control_sign_conventions() -> None:
    """Document the signs the autopilot relies on, measured from the trimmed state."""

    def response(prop: str, delta: float, out: str, steps: int) -> float:
        fdm, _ = make_fdm(BASIC)
        fdm[prop] = fdm[prop] + delta
        for _ in range(steps):
            fdm.run()
        return fdm[out]

    # +elevator -> nose down (negative pitch rate)
    assert response(PROPERTIES["elevator_cmd"], +0.1, PROPERTIES["q_rad_s"], 120) < 0
    assert response(PROPERTIES["elevator_cmd"], -0.1, PROPERTIES["q_rad_s"], 120) > 0
    # +aileron -> roll right (positive roll rate)
    assert response(PROPERTIES["aileron_cmd"], +0.1, PROPERTIES["p_rad_s"], 120) > 0
    # +throttle -> more RPM
    base = response(PROPERTIES["throttle_cmd"], 0.0, PROPERTIES["rpm"], 600)
    assert response(PROPERTIES["throttle_cmd"], +0.1, PROPERTIES["rpm"], 600) > base


def test_autopilot_reaches_and_holds_altitude(basic_flight: pd.DataFrame) -> None:
    df = basic_flight
    # Climb to 6000 ft is commanded at t=0 and should be captured well before t=200.
    hold = df[(df.t_s >= 200) & (df.t_s < 440)]
    assert (hold.altitude_ft - 6000).abs().max() < ALT_TOLERANCE_FT
    # Descent to 5000 ft is commanded at t=440 and must be held over the final minute.
    final = df[df.t_s >= df.t_s.max() - 60]
    assert (final.altitude_ft - 5000).abs().max() < ALT_TOLERANCE_FT


def test_autopilot_captures_headings(basic_flight: pd.DataFrame) -> None:
    df = basic_flight
    for t_check, hdg in [(300, 90.0), (420, 0.0)]:
        window = df[(df.t_s >= t_check) & (df.t_s < t_check + 20)]
        err = (window.heading_deg - hdg).map(wrap_deg).abs().max()
        assert err < HDG_TOLERANCE_DEG, f"heading error {err:.2f} deg at t={t_check}"


def test_log_has_no_nan(basic_flight: pd.DataFrame) -> None:
    sensors = derive_sensors(basic_flight, seed=1)
    assert basic_flight.isna().sum().sum() == 0
    assert sensors.isna().sum().sum() == 0
    assert np.isfinite(basic_flight.to_numpy()).all()


def test_log_rate_is_10hz(basic_flight: pd.DataFrame) -> None:
    dt = np.diff(basic_flight.t_s.to_numpy())
    assert np.allclose(dt, 0.1, atol=1e-6)


def test_reproducible_for_same_seed() -> None:
    short = BASIC.truncated(20)
    a = fly(short).truth
    b = fly(short).truth
    pd.testing.assert_frame_equal(a, b)
    sa = derive_sensors(a, seed=7)
    sb = derive_sensors(b, seed=7)
    pd.testing.assert_frame_equal(sa, sb)
    sc = derive_sensors(a, seed=8)
    assert not sa.drop(columns="t_s").equals(sc.drop(columns="t_s"))


def test_sensors_expose_only_sensor_channels(basic_flight: pd.DataFrame) -> None:
    sensors = derive_sensors(basic_flight, seed=1)
    assert list(sensors.columns) == ["t_s"] + [s.name for s in DEFAULT_SENSORS]
    for col in ("elevator_cmd", "alt_cmd_ft", "lat_deg"):
        assert col not in sensors.columns


def test_sensor_noise_is_bounded(basic_flight: pd.DataFrame) -> None:
    sensors = derive_sensors(basic_flight, seed=1)
    alt_diff = sensors.altitude_ft - basic_flight.altitude_ft
    assert alt_diff.abs().max() < 60  # bias (sd 10) + noise (sd 3) stay well inside this
    assert ((sensors.heading_deg >= 0) & (sensors.heading_deg < 360)).all()


def test_pid_clamps_and_does_not_wind_up() -> None:
    pid = PID(kp=1.0, ki=1.0, out_min=-1.0, out_max=1.0)
    for _ in range(100):
        out = pid.update(10.0, 0.1)
    assert out == 1.0
    assert pid.integral <= 1.0  # frozen once saturated
    # Reversing the error must release saturation immediately, not after a long unwind.
    assert pid.update(-10.0, 0.1) < 0


def test_cli_smoke_flight(tmp_path: Path) -> None:
    """The 20-second smoke flight used by CI."""
    cmd = [
        sys.executable, "-m", "fsfh", "fly",
        "--scenario", "basic", "--seed", "1", "--out", str(tmp_path), "--duration", "20",
    ]  # fmt: skip
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    run_dir = tmp_path / "raw" / "basic-seed1-20s"
    run = read_run(run_dir)
    assert run.meta["seed"] == 1
    assert run.meta["scenario"]["duration_s"] == 20
    assert len(run.truth) == 200
    assert len(run.sensors) == 200
    assert run.truth.isna().sum().sum() == 0
