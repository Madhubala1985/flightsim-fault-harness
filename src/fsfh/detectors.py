"""Fault detectors. They see the sensor frame only, never truth.

Residual detectors (physics consistency, one scalar score per sample):

* ``alt_rate``: slope of the altitude channel over a window vs the vertical
  speed channel. Kinematics: ``d(altitude)/dt`` must equal vertical speed.
* ``airspeed``: slope of the airspeed channel vs a trend predicted from pitch,
  RPM and vertical speed. The structure is physics (speed changes come from
  power and from trading height for speed); the three coefficients are fitted
  by least squares on nominal runs because the aircraft's drag and propeller
  curves are not known to the detector.
* ``energy``: specific energy rate ``V*dV/dt + g*dh/dt`` computed from the
  sensors vs a prediction from RPM and airspeed (power in, drag out), two
  fitted coefficients.
* ``staleness``: seconds a channel has shown exactly the same value (NaN counts
  as not fresh), divided by the allowed staleness (3 s). Its threshold is fixed
  at 1.0, i.e. the rule itself, because quantised nominal channels produce
  short identical streaks by chance and a tuned threshold would chase them.
  Catches a frozen pitot and sensor dropout.

Each residual is divided by its standard deviation on nominal training runs so
all scores are in "nominal sigmas". A detector fires when its score exceeds
its threshold for ``persistence`` consecutive samples. Thresholds are chosen
on *validation* nominal runs to meet a target false-alarm rate (see
``fit_thresholds``).

Data-driven detector: an Isolation Forest on windowed features (mean, std and
slope of each channel over 5 s windows, stride 1 s), trained on nominal runs
only, same thresholding procedure.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from fsfh.autopilot import wrap_deg

HZ = 10
WINDOW_S = 5.0
G_FPS2 = 32.174
FPS_PER_KT = 1.6878
CHANNELS = [
    "airspeed_kts",
    "altitude_ft",
    "vertical_speed_fpm",
    "pitch_deg",
    "roll_deg",
    "heading_deg",
    "rpm",
]
RESIDUAL_DETECTORS = ["alt_rate", "airspeed", "energy", "staleness"]
DETECTORS = RESIDUAL_DETECTORS + ["iforest"]
MODEL_JSON = "detectors.json"
IFOREST_FILE = "iforest.joblib"


# --- signal helpers -----------------------------------------------------------


def rolling_slope(x: np.ndarray, n: int) -> np.ndarray:
    """Least-squares slope per sample over the trailing ``n`` samples (units per sample).

    NaN anywhere in the window gives NaN. The first n-1 samples are NaN.
    """
    k = np.arange(n) - (n - 1) / 2.0
    kernel = k[::-1] / float((k**2).sum())
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1 :] = np.convolve(x, kernel, mode="valid")
    return out


def rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1 :] = np.convolve(x, np.ones(n) / n, mode="valid")
    return out


def stale_seconds(x: np.ndarray, dt: float) -> np.ndarray:
    """Seconds since the channel last changed value. NaN never counts as a change."""
    out = np.zeros(len(x))
    run = 0.0
    for i in range(1, len(x)):
        same = np.isnan(x[i]) or x[i] == x[i - 1]
        run = run + dt if same else 0.0
        out[i] = run
    return out


def residual_inputs(sensors: pd.DataFrame) -> pd.DataFrame:
    """Derived signals shared by the residual detectors, all from sensor channels."""
    n = int(WINDOW_S * HZ)
    dt = 1.0 / HZ
    v_fps = sensors["airspeed_kts"].to_numpy() * FPS_PER_KT
    alt = sensors["altitude_ft"].to_numpy()
    vs_fps = sensors["vertical_speed_fpm"].to_numpy() / 60.0
    out = pd.DataFrame({"t_s": sensors["t_s"].to_numpy()})
    out["alt_slope_fps"] = rolling_slope(alt, n) / dt
    out["vs_mean_fps"] = rolling_mean(vs_fps, n)
    out["v_fps"] = rolling_mean(v_fps, n)
    out["vdot_fps2"] = rolling_slope(v_fps, n) / dt
    out["pitch_deg"] = rolling_mean(sensors["pitch_deg"].to_numpy(), n)
    out["rpm"] = rolling_mean(sensors["rpm"].to_numpy(), n)
    out["energy_rate"] = out["v_fps"] * out["vdot_fps2"] + G_FPS2 * out["vs_mean_fps"]
    return out


def regression_design(inputs: pd.DataFrame, which: str) -> np.ndarray:
    if which == "airspeed":
        cols = ["pitch_deg", "rpm", "vs_mean_fps"]
    elif which == "energy":
        cols = ["rpm", "v_fps"]
    else:
        raise ValueError(which)
    x = inputs[cols].to_numpy()
    return np.column_stack([np.ones(len(x)), x])


def regression_target(inputs: pd.DataFrame, which: str) -> np.ndarray:
    return inputs["vdot_fps2" if which == "airspeed" else "energy_rate"].to_numpy()


# --- model ----------------------------------------------------------------


@dataclass
class DetectorModel:
    """Everything the detectors need at run time, serialised to models/detectors.json."""

    coeffs: dict[str, list[float]] = field(default_factory=dict)
    scales: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    persistence: int = 5
    staleness_s: float = 3.0
    target_fa_per_hour: float = 1.0
    train_runs: list[str] = field(default_factory=list)
    val_runs: list[str] = field(default_factory=list)

    def save(self, model_dir: Path) -> None:
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        (Path(model_dir) / MODEL_JSON).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, model_dir: Path) -> DetectorModel:
        return cls(**json.loads((Path(model_dir) / MODEL_JSON).read_text()))


def residual_scores(sensors: pd.DataFrame, model: DetectorModel) -> pd.DataFrame:
    """Scores in nominal sigmas for the four residual detectors."""
    inp = residual_inputs(sensors)
    out = pd.DataFrame({"t_s": inp["t_s"]})
    out["alt_rate"] = (inp["alt_slope_fps"] - inp["vs_mean_fps"]).abs() / model.scales["alt_rate"]
    for which in ("airspeed", "energy"):
        pred = regression_design(inp, which) @ np.array(model.coeffs[which])
        out[which] = np.abs(regression_target(inp, which) - pred) / model.scales[which]
    stale = np.column_stack(
        [stale_seconds(sensors[c].to_numpy(dtype=float), 1.0 / HZ) for c in CHANNELS]
    )
    out["staleness"] = stale.max(axis=1) / model.staleness_s
    return out


# --- isolation forest -------------------------------------------------------


def window_features(sensors: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-window features (mean, std, slope per channel), stride 1 s. Returns (t_end, X)."""
    n = int(WINDOW_S * HZ)
    stride = HZ
    t = sensors["t_s"].to_numpy()
    data = sensors[CHANNELS].to_numpy(dtype=float).copy()
    # heading: use turn rate (wrapped diff) instead of the raw angle, which wraps at 360
    hdg_idx = CHANNELS.index("heading_deg")
    hdg = data[:, hdg_idx]
    turn = np.r_[0.0, [wrap_deg(d) for d in np.diff(hdg)]] * HZ
    data[:, hdg_idx] = turn
    ends = np.arange(n - 1, len(t), stride)
    rows = []
    for e in ends:
        w = data[e - n + 1 : e + 1]
        k = np.arange(n) - (n - 1) / 2.0
        slope = (k[:, None] * (w - w.mean(axis=0))).sum(axis=0) / (k**2).sum()
        rows.append(np.r_[w.mean(axis=0), w.std(axis=0), slope])
    return t[ends], np.array(rows)


def iforest_scores(sensors: pd.DataFrame, forest: IsolationForest) -> pd.DataFrame:
    """Anomaly score (higher = stranger) per sample, held from the last complete window."""
    t_end, x = window_features(sensors)
    score = np.full(len(x), np.nan)
    ok = ~np.isnan(x).any(axis=1)
    if ok.any():
        score[ok] = -forest.decision_function(x[ok])
    t = sensors["t_s"].to_numpy()
    idx = np.searchsorted(t_end, t, side="right") - 1
    held = np.where(idx >= 0, score[np.clip(idx, 0, len(score) - 1)], np.nan)
    return pd.DataFrame({"t_s": t, "iforest": held})


# --- running everything -----------------------------------------------------


def score_all(sensors: pd.DataFrame, model: DetectorModel, forest: IsolationForest) -> pd.DataFrame:
    scores = residual_scores(sensors, model)
    scores["iforest"] = iforest_scores(sensors, forest)["iforest"].to_numpy()
    return scores


def alarms_from_scores(scores: pd.DataFrame, model: DetectorModel) -> pd.DataFrame:
    """Boolean alarm per detector: score above threshold for ``persistence`` samples in a row."""
    out = pd.DataFrame({"t_s": scores["t_s"]})
    for name in DETECTORS:
        above = (scores[name].to_numpy() > model.thresholds[name]).astype(float)
        run = np.convolve(above, np.ones(model.persistence), mode="full")[: len(above)]
        out[name] = run >= model.persistence
    out["any"] = out[DETECTORS].any(axis=1)
    return out


def alarm_events(alarm: np.ndarray, t: np.ndarray) -> list[float]:
    """Times at which an alarm turns on (rising edges)."""
    a = alarm.astype(bool)
    rising = np.flatnonzero(a & ~np.r_[False, a[:-1]])
    return [float(t[i]) for i in rising]


def detect(sensors: pd.DataFrame, model_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: load models, return (scores, alarms) for one sensor frame."""
    model = DetectorModel.load(model_dir)
    forest = joblib.load(Path(model_dir) / IFOREST_FILE)
    scores = score_all(sensors, model, forest)
    return scores, alarms_from_scores(scores, model)


# --- fitting ----------------------------------------------------------------


def fit_thresholds(
    scores_by_run: dict[str, pd.DataFrame], model: DetectorModel
) -> dict[str, float]:
    """Lowest threshold per detector whose alarm-event rate on validation runs is
    at or below ``target_fa_per_hour``. Candidates are quantiles of the pooled score."""
    hours = sum(len(s) for s in scores_by_run.values()) / HZ / 3600.0
    # staleness is a rule, not a statistic: "unchanged for staleness_s" is the threshold itself
    thresholds = {"staleness": 1.0}
    for name in [d for d in DETECTORS if d != "staleness"]:
        pooled = np.concatenate([s[name].dropna().to_numpy() for s in scores_by_run.values()])
        candidates = np.unique(np.quantile(pooled, np.linspace(0.90, 1.0, 201)))
        chosen = float(candidates[-1]) * 1.05  # fallback: above everything seen
        for thr in candidates:
            events = 0
            for s in scores_by_run.values():
                above = (s[name].to_numpy() > thr).astype(float)
                run = np.convolve(above, np.ones(model.persistence), mode="full")[: len(above)]
                events += len(alarm_events(run >= model.persistence, s["t_s"].to_numpy()))
            if events / hours <= model.target_fa_per_hour:
                chosen = float(thr)
                break
        thresholds[name] = chosen
    return thresholds


def fit(
    nominal_sensors: dict[str, pd.DataFrame],
    model_dir: Path,
    target_fa_per_hour: float = 1.0,
    val_fraction: float = 0.3,
    holdout_fraction: float = 0.3,
    seed: int = 0,
) -> DetectorModel:
    """Fit coefficients, scales and the forest on training runs; thresholds on validation.

    The last ``holdout_fraction`` of nominal runs (by sorted id) are touched by
    neither step; ``fsfh.evaluate`` measures false alarms on those only.
    """
    ids = sorted(nominal_sensors)
    n_val = max(1, int(round(val_fraction * len(ids))))
    n_hold = max(1, int(round(holdout_fraction * len(ids))))
    val_ids, train_ids = ids[:n_val], ids[n_val : len(ids) - n_hold]
    model = DetectorModel(target_fa_per_hour=target_fa_per_hour)
    model.train_runs, model.val_runs = train_ids, val_ids

    train_inputs = pd.concat([residual_inputs(nominal_sensors[i]) for i in train_ids]).dropna()
    for which in ("airspeed", "energy"):
        x, y = regression_design(train_inputs, which), regression_target(train_inputs, which)
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        model.coeffs[which] = [float(c) for c in coef]
        model.scales[which] = float(np.std(y - x @ coef))
    model.scales["alt_rate"] = float(
        np.std(train_inputs["alt_slope_fps"] - train_inputs["vs_mean_fps"])
    )

    x_train = np.vstack([window_features(nominal_sensors[i])[1] for i in train_ids])
    forest = IsolationForest(n_estimators=200, contamination="auto", random_state=seed)
    forest.fit(x_train)

    model.thresholds = {name: np.inf for name in DETECTORS}  # placeholder for score_all
    val_scores = {i: score_all(nominal_sensors[i], model, forest) for i in val_ids}
    model.thresholds = fit_thresholds(val_scores, model)
    model.save(model_dir)
    joblib.dump(forest, Path(model_dir) / IFOREST_FILE)
    return model
