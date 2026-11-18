"""Plot truth vs sensor channels for one run, with fault windows shaded."""

from __future__ import annotations

from pathlib import Path

from fsfh.autopilot import wrap_deg
from fsfh.runlog import read_run

PAIRS = [  # (sensor column, truth column, label)
    ("airspeed_kts", "ias_kts", "airspeed [kts]"),
    ("altitude_ft", "altitude_ft", "altitude [ft]"),
    ("vertical_speed_fpm", "vs_fpm", "vertical speed [fpm]"),
    ("pitch_deg", "pitch_deg", "pitch [deg]"),
    ("roll_deg", "roll_deg", "roll [deg]"),
    ("heading_deg", "heading_deg", "heading [deg, +/-180]"),
    ("rpm", "rpm", "engine [rpm]"),
]


def plot_run(run_dir: Path, out_png: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = read_run(run_dir)
    fig, axes = plt.subplots(len(PAIRS), 1, figsize=(11, 2.0 * len(PAIRS)), sharex=True)
    for ax, (s_col, t_col, label) in zip(axes, PAIRS, strict=True):
        truth_y, sensor_y = run.truth[t_col], run.sensors[s_col]
        if s_col == "heading_deg":  # avoid 0/360 wrap lines on a 360 heading
            truth_y, sensor_y = truth_y.map(wrap_deg), sensor_y.map(wrap_deg)
        ax.plot(run.truth.t_s, truth_y, color="black", lw=1.0, label="truth")
        ax.plot(run.sensors.t_s, sensor_y, color="tab:red", lw=0.6, alpha=0.8, label="sensor")
        for f in run.faults:
            ax.axvspan(f.onset_s, f.end_s, color="tab:orange", alpha=0.2)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right")
    faults = ", ".join(
        f"{f.type} t={f.onset_s:.0f}-{f.end_s:.0f}s mag={f.magnitude}" for f in run.faults
    )
    fig.suptitle(f"{run.meta['run_id']}  {faults or 'nominal'}")
    axes[-1].set_xlabel("t [s]")
    fig.tight_layout()
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png
