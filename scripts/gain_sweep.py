"""How the autopilot gains were chosen: a small grid sweep scored on the basic scenario.

Run: .venv/bin/python scripts/gain_sweep.py
Each candidate flies the full scenario (~2 s). The score is altitude RMS during the
hold windows; vertical-speed jitter and capture time are printed alongside so a
candidate that holds altitude by oscillating is visible.
"""

from __future__ import annotations

import itertools

import numpy as np

from fsfh.autopilot import AutopilotGains
from fsfh.scenarios import BASIC
from fsfh.sim import fly


def score(gains: AutopilotGains) -> dict[str, float]:
    df = fly(BASIC, gains).truth
    hold = df[(df.t_s >= 200) & (df.t_s < 440)]
    err = hold.altitude_ft - hold.alt_cmd_ft
    captured = df[(df.altitude_ft - 6000).abs() < 30]
    return {
        "alt_rms_ft": float(np.sqrt((err**2).mean())),
        "alt_max_ft": float(err.abs().max()),
        "vs_std_fpm": float(hold.vs_fpm.std()),
        "capture_s": float(captured.t_s.iloc[0]) if len(captured) else float("inf"),
    }


def main() -> None:
    grid = itertools.product([0.03, 0.06, 0.08], [0.004, 0.01], [1.0, 2.0], [1.0, 3.0])
    rows = []
    for vs_kp, vs_ki, damping, alt_to_vs in grid:
        gains = AutopilotGains(
            vs_kp=vs_kp, vs_ki=vs_ki, pitch_rate_damping=damping, alt_to_vs=alt_to_vs
        )
        rows.append(((vs_kp, vs_ki, damping, alt_to_vs), score(gains)))
    rows.sort(key=lambda r: r[1]["alt_rms_ft"])
    print("(vs_kp, vs_ki, pitch_rate_damping, alt_to_vs)")
    for params, s in rows:
        print(params, {k: round(v, 1) for k, v in s.items()})


if __name__ == "__main__":
    main()
