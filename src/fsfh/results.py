"""Summarise runs into a Markdown table and splice it into the README.

This is the only way numbers get into the README: ``make results`` regenerates
the table from whatever is under ``runs/raw/``. Nothing is typed by hand.

Per segment, "captured" means the first sample at which altitude is within
``CAPTURE_ALT_FT`` of target and heading within ``CAPTURE_HDG_DEG``. Hold
metrics are computed from that sample to the end of the segment. The reported
capture time is the worst segment. A segment that never captures (for example
a 20 s smoke run) contributes no hold samples and is counted in ``uncaptured``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from fsfh.autopilot import wrap_deg
from fsfh.runlog import RunData, list_runs, read_run

CAPTURE_ALT_FT = 30.0
CAPTURE_HDG_DEG = 3.0
README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"
EVAL_START = "<!-- eval:start -->"
EVAL_END = "<!-- eval:end -->"


def held_mask(
    truth: pd.DataFrame, segments: list[dict], duration_s: float
) -> tuple[np.ndarray, float, int]:
    """Mask of samples after capture in each segment, worst capture time, uncaptured count."""
    t = truth["t_s"].to_numpy()
    alt_err = (truth["altitude_ft"] - truth["alt_cmd_ft"]).abs().to_numpy()
    hdg_err = (truth["heading_deg"] - truth["heading_cmd_deg"]).map(wrap_deg).abs().to_numpy()
    captured = (alt_err < CAPTURE_ALT_FT) & (hdg_err < CAPTURE_HDG_DEG)
    mask = np.zeros(len(t), dtype=bool)
    worst_capture_s = 0.0
    uncaptured = 0
    starts = [s["start_s"] for s in segments] + [duration_s]
    for a, b in zip(starts[:-1], starts[1:], strict=True):
        in_seg = (t >= a) & (t < b)
        hits = np.flatnonzero(in_seg & captured)
        if len(hits) == 0:
            uncaptured += 1
            continue
        first = hits[0]
        mask |= in_seg & (np.arange(len(t)) >= first)
        worst_capture_s = max(worst_capture_s, float(t[first] - a))
    return mask, worst_capture_s, uncaptured


def summarise(run: RunData) -> dict:
    truth = run.truth
    scenario = run.meta["scenario"]
    m, worst_capture_s, uncaptured = held_mask(truth, scenario["segments"], scenario["duration_s"])
    s = truth[m]
    alt_err = s["altitude_ft"] - s["alt_cmd_ft"]
    hdg_err = (s["heading_deg"] - s["heading_cmd_deg"]).map(wrap_deg)
    ias_err = s["ias_kts"] - s["ias_cmd_kts"]
    rms = lambda e: float(np.sqrt(np.mean(e**2))) if len(e) else float("nan")  # noqa: E731
    return {
        "run": run.meta["run_id"],
        "seed": run.meta["seed"],
        "duration_s": scenario["duration_s"],
        "samples": len(truth),
        "held_samples": int(m.sum()),
        "uncaptured_segments": uncaptured,
        "worst_capture_s": worst_capture_s,
        "alt_rms_ft": rms(alt_err),
        "alt_max_abs_ft": float(alt_err.abs().max()) if len(s) else float("nan"),
        "hdg_rms_deg": rms(hdg_err),
        "ias_rms_kts": rms(ias_err),
        "max_bank_deg": float(truth["roll_deg"].abs().max()),
        "nan_count": int(truth.isna().sum().sum() + run.sensors.isna().sum().sum()),
    }


def results_table(runs_dir: Path) -> str:
    rows = [summarise(read_run(p)) for p in list_runs(runs_dir)]
    if not rows:
        return "_No runs found under `runs/raw/`. Run `make fly` first._\n"
    df = pd.DataFrame(rows)
    lines = [
        "| run | seed | duration s | worst capture s | uncaptured segs | alt RMS ft "
        "| alt max ft | hdg RMS deg | IAS RMS kts | max bank deg | NaN |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in df.itertuples(index=False):
        lines.append(
            f"| {r.run} | {r.seed} | {r.duration_s:.0f} | {r.worst_capture_s:.1f} | "
            f"{r.uncaptured_segments} | {r.alt_rms_ft:.1f} | {r.alt_max_abs_ft:.1f} | "
            f"{r.hdg_rms_deg:.2f} | {r.ias_rms_kts:.2f} | {r.max_bank_deg:.1f} | {r.nan_count} |"
        )
    lines.append("")
    lines.append(
        f"Capture = altitude within {CAPTURE_ALT_FT:.0f} ft and heading within "
        f"{CAPTURE_HDG_DEG:.0f} deg of target. Hold metrics (alt, hdg, IAS) are over the "
        "samples after capture in each segment."
    )
    return "\n".join(lines) + "\n"


def splice_readme(
    readme: Path, body: str, start: str = README_START, end: str = README_END
) -> None:
    text = readme.read_text()
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        return  # README does not carry this table (the per-run table lives in results/runs.md)
    replacement = f"{start}\n{body}{end}"
    readme.write_text(pattern.sub(lambda _: replacement, text))


def make_results(
    runs_dir: Path,
    readme: Path | None,
    out_dir: Path = Path("results"),
    manifest: Path | None = None,
    model_dir: Path | None = None,
    triage: tuple[Path, str, str | None] | None = None,
) -> str:
    """Regenerate results/*.md (and the README tables) from runs/raw and the models."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = results_table(runs_dir)
    (out_dir / "runs.md").write_text(table)
    if readme is not None:
        splice_readme(readme, table)
    if manifest is not None and model_dir is not None and Path(manifest).is_file():
        from fsfh.evaluate import evaluate, write_reports

        reports = write_reports(evaluate(manifest, model_dir), out_dir)
        body = reports["evaluation"] + "\n" + reports["confusion"]
        body += "\n![confusion](results/confusion.png)\n"
        if triage is not None and Path(triage[0]).is_dir():
            body += "\n" + triage_section(manifest, model_dir, out_dir, *triage)
        table += "\n" + body
        if readme is not None:
            splice_readme(readme, body, EVAL_START, EVAL_END)
    return table


def triage_section(
    manifest: Path, model_dir: Path, out_dir: Path, cache: Path, backend: str, model: str | None
) -> str:
    """Triage score from cached LLM responses only. Never calls the network."""
    from fsfh.triage.backends import make_backend
    from fsfh.triage.evaluate import triage_campaign, write_report

    cached = make_backend(backend, model, cache, offline=True)
    try:
        return write_report(triage_campaign(manifest, model_dir, cached), out_dir, cached)
    except FileNotFoundError as e:
        return f"_Triage section skipped: {e}. Run `make triage` with an API key first._\n"
