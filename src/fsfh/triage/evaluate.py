"""Score the triage agent against injected ground truth on a campaign.

Windows come only from alarms (never from the fault log): one window per
alarm rising edge, plus two fixed windows per nominal run (t = 250 and 400 s,
level flight) for the "unknown on nominal" rate. The label of a window is the
fault type if the fault was active anywhere inside it, else "nominal". Faulted
runs that raised no alarm produce no window; they are counted as "untriaged".
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from fsfh.runlog import read_run
from fsfh.triage.agent import WINDOW_AFTER_S, WINDOW_BEFORE_S, triage_run
from fsfh.triage.backends import CachedBackend

NOMINAL_WINDOW_STARTS = (250.0, 400.0)
CALIBRATION_BINS = (0.0, 0.5, 0.7, 0.9, 1.01)


def label_for(row: dict, t0: float, t1: float) -> str:
    if row["fault_type"] == "nominal":
        return "nominal"
    onset, end = float(row["onset_s"]), float(row["onset_s"]) + float(row["duration_s"])
    return row["fault_type"] if onset < t1 and end > t0 else "nominal"


def triage_campaign(manifest: Path, model_dir: Path, backend: CachedBackend) -> pd.DataFrame:
    rows = []
    untriaged = 0
    for r in csv.DictReader(Path(manifest).open()):
        run = read_run(Path(r["path"]))
        windows = None
        if r["fault_type"] == "nominal":
            windows = [(t, t + WINDOW_BEFORE_S + WINDOW_AFTER_S) for t in NOMINAL_WINDOW_STARTS]
        results = triage_run(run.sensors, model_dir, backend, windows)
        if not results:
            untriaged += 1
        for t0, t1, rep in results:
            label = label_for(r, t0, t1)
            rows.append(
                {
                    "run_id": r["run_id"],
                    "t0": t0,
                    "t1": t1,
                    "label": label,
                    "predicted": rep.fault_type,
                    "confidence": rep.confidence,
                    "correct": rep.fault_type == label,
                    "recommended_check": rep.recommended_check,
                }
            )
    df = pd.DataFrame(rows)
    df.attrs["untriaged_faulted_runs"] = untriaged
    return df


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, g in df[df.label != "nominal"].groupby("label"):
        rows.append(
            {
                "fault_type": label,
                "windows": len(g),
                "accuracy": g.correct.mean(),
                "said_unknown": (g.predicted == "unknown").mean(),
                "mean_confidence": g.confidence.mean(),
            }
        )
    return pd.DataFrame(rows)


def nominal_table(df: pd.DataFrame) -> dict:
    g = df[df.label == "nominal"]
    return {
        "windows": len(g),
        "said_unknown": float((g.predicted == "unknown").mean()) if len(g) else np.nan,
        "false_fault_rate": float((g.predicted != "unknown").mean()) if len(g) else np.nan,
    }


def calibration_table(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Correctness per confidence bin on windows where a fault was named, plus ECE."""
    named = df[df.predicted != "unknown"]
    rows, ece = [], 0.0
    for lo, hi in zip(CALIBRATION_BINS[:-1], CALIBRATION_BINS[1:], strict=True):
        g = named[(named.confidence >= lo) & (named.confidence < hi)]
        if len(g) == 0:
            continue
        acc, conf = g.correct.mean(), g.confidence.mean()
        ece += len(g) / len(named) * abs(acc - conf)
        rows.append({"confidence_bin": f"[{lo:.1f}, {min(hi, 1.0):.1f})", "windows": len(g),
                     "mean_confidence": conf, "accuracy": acc})  # fmt: skip
    return pd.DataFrame(rows), float(ece)


def write_report(df: pd.DataFrame, out_dir: Path, backend: CachedBackend) -> str:
    from fsfh.evaluate import md_table

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "triage_windows.csv", index=False)
    nom = nominal_table(df)
    calib, ece = calibration_table(df)
    overall = df[df.label != "nominal"].correct.mean() if (df.label != "nominal").any() else np.nan
    text = (
        f"### Triage agent ({backend.id} / {backend.model}, {len(df)} windows, "
        f"served from cache: {backend.hits}, new calls: {backend.misses})\n\n"
        "The agent sees only the alarm list and a sensor feature summary for each window, "
        "never truth or the fault log. Windows are chosen from alarms alone; the label is "
        "assigned afterwards from the fault log.\n\n"
        f"Overall accuracy on fault windows: {overall:.2f}. Faulted runs with no alarm "
        f"(untriaged): {df.attrs.get('untriaged_faulted_runs', 0)}.\n\n"
        + md_table(accuracy_table(df))
        + f"\nNominal windows: {nom['windows']}; said 'unknown' on {nom['said_unknown']:.2f} "
        f"of them, named a fault on {nom['false_fault_rate']:.2f}.\n\n"
        f"Calibration (expected calibration error {ece:.3f}):\n\n" + md_table(calib)
    )
    (out_dir / "triage.md").write_text(text)
    return text
