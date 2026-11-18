"""Derive noisy "sensor" channels from the truth log.

Detectors built in later phases only ever see the output of this module,
never the truth frame. Each channel is truth passed through, in order:

1. a first-order lag (only the vertical speed indicator uses this; a real VSI
   is a pressure-leak instrument that lags by a few seconds),
2. a constant per-run bias drawn once from N(0, bias_sd),
3. white noise drawn per sample from N(0, noise_sd),
4. quantisation to the instrument's display resolution,
5. optional wrap into [0, 360) for heading.

All randomness comes from one ``numpy.random.default_rng(seed)``, so the same
seed and the same truth give byte-identical sensor channels.

The noise magnitudes are plausible round numbers for light-aircraft
instruments, not values taken from any instrument specification. See the
README "Limitations" section.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SensorSpec:
    name: str  # output column name
    source: str  # truth column name
    noise_sd: float  # white noise standard deviation, output units
    bias_sd: float  # standard deviation of the per-run constant bias
    quant: float  # display resolution; 0 disables quantisation
    lag_s: float = 0.0  # first-order lag time constant; 0 disables the lag
    wrap_360: bool = False


DEFAULT_SENSORS: tuple[SensorSpec, ...] = (
    SensorSpec("airspeed_kts", "ias_kts", noise_sd=0.5, bias_sd=1.0, quant=0.1),
    SensorSpec("altitude_ft", "altitude_ft", noise_sd=3.0, bias_sd=10.0, quant=1.0),
    SensorSpec("vertical_speed_fpm", "vs_fpm", noise_sd=20.0, bias_sd=10.0, quant=10.0, lag_s=2.0),
    SensorSpec("pitch_deg", "pitch_deg", noise_sd=0.1, bias_sd=0.2, quant=0.01),
    SensorSpec("roll_deg", "roll_deg", noise_sd=0.1, bias_sd=0.2, quant=0.01),
    SensorSpec("heading_deg", "heading_deg", noise_sd=0.3, bias_sd=0.5, quant=0.1, wrap_360=True),
    SensorSpec("rpm", "rpm", noise_sd=5.0, bias_sd=5.0, quant=10.0),
)


def first_order_lag(x: np.ndarray, tau_s: float, dt_s: float) -> np.ndarray:
    """Discrete first-order low-pass: y[k] = y[k-1] + a * (x[k] - y[k-1])."""
    if tau_s <= 0:
        return x.copy()
    alpha = dt_s / (tau_s + dt_s)
    y = np.empty_like(x)
    y[0] = x[0]
    for k in range(1, len(x)):
        y[k] = y[k - 1] + alpha * (x[k] - y[k - 1])
    return y


def derive_sensors(
    truth: pd.DataFrame,
    seed: int,
    specs: tuple[SensorSpec, ...] = DEFAULT_SENSORS,
) -> pd.DataFrame:
    """Build the sensor frame from a truth frame. Columns: t_s plus one per spec."""
    rng = np.random.default_rng(seed)
    t = truth["t_s"].to_numpy()
    dt_s = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    out: dict[str, np.ndarray] = {"t_s": t}
    for spec in specs:
        x = truth[spec.source].to_numpy(dtype=np.float64)
        y = first_order_lag(x, spec.lag_s, dt_s)
        y = y + rng.normal(0.0, spec.bias_sd) + rng.normal(0.0, spec.noise_sd, size=len(y))
        if spec.quant > 0:
            y = np.round(y / spec.quant) * spec.quant
        if spec.wrap_360:
            y = np.mod(y, 360.0)
        out[spec.name] = y
    return pd.DataFrame(out)


def specs_to_dicts(specs: tuple[SensorSpec, ...] = DEFAULT_SENSORS) -> list[dict]:
    return [asdict(s) for s in specs]
