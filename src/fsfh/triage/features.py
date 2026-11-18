"""Feature summary of the sensor channels in a window, as plain text for the agent."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fsfh.detectors import CHANNELS, DETECTORS, alarm_events


def window_slice(frame: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
    return frame[(frame.t_s >= t0) & (frame.t_s < t1)]


def longest_constant_run(x: np.ndarray) -> int:
    """Longest run of consecutive identical values (NaN counts as identical)."""
    best = run = 1
    for i in range(1, len(x)):
        same = (np.isnan(x[i]) and np.isnan(x[i - 1])) or x[i] == x[i - 1]
        run = run + 1 if same else 1
        best = max(best, run)
    return best if len(x) else 0


def channel_summary(sensors: pd.DataFrame, t0: float, t1: float) -> list[dict]:
    w = window_slice(sensors, t0, t1)
    rows = []
    for c in CHANNELS:
        x = w[c].to_numpy(dtype=float)
        ok = ~np.isnan(x)
        if ok.sum() < 2:
            rows.append({"channel": c, "nan_fraction": 1.0})
            continue
        if c == "heading_deg":  # unwrap so stats are continuous across 0/360
            x = np.where(ok, np.degrees(np.unwrap(np.radians(np.nan_to_num(x)))), np.nan)
        t = w.t_s.to_numpy()[ok]
        slope = float(np.polyfit(t - t[0], x[ok], 1)[0])
        rows.append(
            {
                "channel": c,
                "mean": round(float(x[ok].mean()), 2),
                "std": round(float(x[ok].std()), 3),
                "min": round(float(x[ok].min()), 2),
                "max": round(float(x[ok].max()), 2),
                "slope_per_s": round(slope, 4),
                "distinct_values": int(len(np.unique(x[ok]))),
                "longest_constant_s": round(longest_constant_run(x) / 10.0, 1),
                "nan_fraction": round(float(1 - ok.mean()), 3),
            }
        )
    return rows


def alarm_summary(scores: pd.DataFrame, alarms: pd.DataFrame, t0: float, t1: float) -> list[dict]:
    w_alarm = window_slice(alarms, t0, t1)
    w_score = window_slice(scores, t0, t1)
    out = []
    for name in DETECTORS:
        a = w_alarm[name].to_numpy()
        if not a.any():
            continue
        onsets = alarm_events(a, w_alarm.t_s.to_numpy())
        out.append(
            {
                "detector": name,
                "first_alarm_t_s": round(onsets[0], 1) if onsets else round(float(t0), 1),
                "active_fraction": round(float(a.mean()), 2),
                "peak_score": round(float(np.nanmax(w_score[name].to_numpy())), 2),
            }
        )
    return out


def format_window(t0: float, t1: float, alarms: list[dict], channels: list[dict]) -> str:
    lines = [f"Window: t = {t0:.1f} s to {t1:.1f} s (10 Hz sensor data).", "", "Alarms:"]
    if not alarms:
        lines.append("  (none)")
    for a in alarms:
        lines.append(
            f"  - {a['detector']}: first at t={a['first_alarm_t_s']} s, active "
            f"{a['active_fraction'] * 100:.0f}% of window, "
            f"peak score {a['peak_score']} nominal-sigma"
        )
    lines += ["", "Sensor channel summary over the window:"]
    for c in channels:
        if c.get("nan_fraction", 0) >= 1.0:
            lines.append(f"  - {c['channel']}: all NaN")
            continue
        lines.append(
            f"  - {c['channel']}: mean {c['mean']}, std {c['std']}, "
            f"range [{c['min']}, {c['max']}], "
            f"slope {c['slope_per_s']}/s, {c['distinct_values']} distinct values, "
            f"longest constant stretch {c['longest_constant_s']} s, "
            f"NaN fraction {c['nan_fraction']}"
        )
    return "\n".join(lines)
