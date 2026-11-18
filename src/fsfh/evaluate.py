"""Score every run in a campaign manifest and summarise detector performance.

Definitions:

* A faulted run is **detected** by a detector if that detector has an alarm
  event (rising edge) in ``[onset_s, end_s + GRACE_S)``. Time-to-detect is the
  first such event minus onset.
* **False alarms per flight hour** are alarm events on *held-out* nominal runs:
  nominal runs that were used neither to train nor to validate the model.
  Alarm events before onset on faulted runs are reported as a separate count.
* The confusion table is fault type x detector -> fraction of runs of that
  type the detector detected.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from fsfh.detectors import DETECTORS, alarm_events, detect
from fsfh.runlog import read_run

GRACE_S = 10.0
ALL = DETECTORS + ["any"]


def events_after(events: list[float], onset: float, end: float) -> float | None:
    hits = [e for e in events if onset <= e < end + GRACE_S]
    return hits[0] - onset if hits else None


def score_run(row: dict, model_dir: Path, exclude: set[str]) -> dict:
    run = read_run(Path(row["path"]))
    _, alarms = detect(run.sensors, model_dir)
    t = alarms["t_s"].to_numpy()
    hours = len(t) / 10 / 3600.0
    out = {
        "run_id": row["run_id"],
        "fault_type": row["fault_type"],
        "held_out": row["fault_type"] == "nominal" and row["run_id"] not in exclude,
        "hours": hours,
    }
    for name in ALL:
        ev = alarm_events(alarms[name].to_numpy(), t)
        if row["fault_type"] == "nominal":
            out[f"{name}_events"] = len(ev)
            out[f"{name}_ttd"] = np.nan
        else:
            onset, end = float(row["onset_s"]), float(row["onset_s"]) + float(row["duration_s"])
            ttd = events_after(ev, onset, end)
            out[f"{name}_ttd"] = np.nan if ttd is None else ttd
            out[f"{name}_events"] = sum(e < onset for e in ev)  # pre-onset false alarms
    return out


def evaluate(manifest: Path, model_dir: Path) -> pd.DataFrame:
    from fsfh.detectors import DetectorModel

    model = DetectorModel.load(model_dir)
    exclude = set(model.train_runs) | set(model.val_runs)
    rows = list(csv.DictReader(Path(manifest).open()))
    return pd.DataFrame([score_run(r, model_dir, exclude) for r in rows])


def per_fault_table(df: pd.DataFrame) -> pd.DataFrame:
    faulted = df[df.fault_type != "nominal"]
    rows = []
    for ftype, g in faulted.groupby("fault_type"):
        det = g["any_ttd"].notna()
        rows.append(
            {
                "fault_type": ftype,
                "runs": len(g),
                "detection_rate": det.mean(),
                "median_ttd_s": g.loc[det, "any_ttd"].median() if det.any() else np.nan,
                "pre_onset_false_alarms": int(g["any_events"].sum()),
            }
        )
    return pd.DataFrame(rows)


def false_alarm_table(df: pd.DataFrame) -> pd.DataFrame:
    held = df[df.held_out]
    hours = held["hours"].sum()
    rows = [
        {"detector": name, "events": int(held[f"{name}_events"].sum()),
         "per_hour": held[f"{name}_events"].sum() / hours if hours else np.nan}
        for name in ALL
    ]  # fmt: skip
    return pd.DataFrame(rows), hours, len(held)


def confusion_table(df: pd.DataFrame) -> pd.DataFrame:
    faulted = df[df.fault_type != "nominal"]
    conf = {}
    for name in ALL:
        conf[name] = faulted.groupby("fault_type")[f"{name}_ttd"].apply(lambda s: s.notna().mean())
    return pd.DataFrame(conf)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_reports(df: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fa, hours, n_held = false_alarm_table(df)
    per_fault = per_fault_table(df)
    conf = confusion_table(df).reset_index()
    n_faulted = int((df.fault_type != "nominal").sum())

    evaluation = (
        f"### Detection per fault type ({n_faulted} faulted runs, any detector)\n\n"
        + md_table(per_fault)
        + f"\nTime-to-detect is measured from fault onset; a run counts as detected if an alarm "
        f"starts between onset and {GRACE_S:.0f} s after the fault ends.\n\n"
        f"### False alarms on held-out nominal runs ({n_held} runs, {hours:.2f} flight hours)\n\n"
        + md_table(fa)
        + "\nHeld-out = nominal runs used neither for training nor for threshold validation.\n"
    )
    confusion = (
        "### Which detector fires for which fault (fraction of runs detected)\n\n" + md_table(conf)
    )
    (out_dir / "evaluation.md").write_text(evaluation)
    (out_dir / "confusion.md").write_text(confusion)
    df.to_csv(out_dir / "per_run.csv", index=False)
    plot_confusion(conf.set_index("fault_type"), out_dir / "confusion.png")
    return {"evaluation": evaluation, "confusion": confusion}


def plot_confusion(conf: pd.DataFrame, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(conf.to_numpy(dtype=float), vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(conf.columns)), conf.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(conf.index)), conf.index)
    for i in range(len(conf.index)):
        for j in range(len(conf.columns)):
            v = conf.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.6 else "black", fontsize=8)  # fmt: skip
    fig.colorbar(im, ax=ax, label="fraction of runs detected")
    ax.set_title("Detector x fault type")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
