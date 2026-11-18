"""The structured report the agent must return, validated with pydantic."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from fsfh.faults import FAULT_TYPES

FaultLabel = Literal[
    "stuck_airspeed",
    "altimeter_bias",
    "altimeter_drift",
    "attitude_noise_burst",
    "sensor_dropout",
    "engine_power_loss",
    "control_lag",
    "unknown",
]
assert set(FaultLabel.__args__) == set(FAULT_TYPES) | {"unknown"}  # keep in sync with faults.py


class TriageReport(BaseModel):
    """What the triage agent returns for one flagged window."""

    fault_type: FaultLabel = Field(description="Most likely fault, or 'unknown' if unsure/nominal")
    confidence: float = Field(ge=0.0, le=1.0, description="Probability that fault_type is right")
    evidence: list[str] = Field(
        min_length=1, description="Quotes of the alarms or features that drove the decision"
    )
    recommended_check: str = Field(description="One concrete next check for a test engineer")
