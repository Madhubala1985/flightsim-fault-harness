"""Scripted flight scenarios: an initial condition plus a timeline of targets.

A scenario is pure data. ``targets_at(t)`` returns the active targets for a
given sim time, taking the last segment whose start time is <= t. Step changes
in targets are fine because the autopilot's outer loops clamp vertical speed
and bank angle, which turns a step into a smooth capture.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from fsfh.autopilot import Targets


@dataclass(frozen=True)
class Segment:
    """Targets that become active at ``start_s`` and stay until the next segment."""

    start_s: float
    altitude_ft: float
    heading_deg: float
    airspeed_kts: float


@dataclass(frozen=True)
class Scenario:
    name: str
    aircraft: str
    duration_s: float
    initial_altitude_ft: float
    initial_heading_deg: float
    initial_airspeed_kts: float
    segments: tuple[Segment, ...] = field(default_factory=tuple)

    def targets_at(self, t: float) -> Targets:
        active = self.segments[0]
        for seg in self.segments:
            if seg.start_s <= t:
                active = seg
        return Targets(
            altitude_ft=active.altitude_ft,
            heading_deg=active.heading_deg,
            airspeed_kts=active.airspeed_kts,
        )

    def truncated(self, duration_s: float) -> Scenario:
        """Same scenario, shorter run. Used for smoke tests."""
        return Scenario(**{**asdict(self), "duration_s": duration_s, "segments": self.segments})

    def to_dict(self) -> dict:
        return asdict(self)


BASIC = Scenario(
    name="basic",
    aircraft="c172x",
    duration_s=660.0,
    initial_altitude_ft=5000.0,
    initial_heading_deg=0.0,
    initial_airspeed_kts=100.0,
    segments=(
        # Climb from 5000 to 6000 ft, then hold.
        Segment(start_s=0.0, altitude_ft=6000.0, heading_deg=0.0, airspeed_kts=100.0),
        # First heading change: right turn to 090.
        Segment(start_s=200.0, altitude_ft=6000.0, heading_deg=90.0, airspeed_kts=100.0),
        # Second heading change: left turn back to 360.
        Segment(start_s=320.0, altitude_ft=6000.0, heading_deg=0.0, airspeed_kts=100.0),
        # Descend back to 5000 ft and hold until the end.
        Segment(start_s=440.0, altitude_ft=5000.0, heading_deg=0.0, airspeed_kts=100.0),
    ),
)

SCENARIOS: dict[str, Scenario] = {BASIC.name: BASIC}


def get_scenario(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError:
        known = ", ".join(sorted(SCENARIOS))
        raise KeyError(f"unknown scenario {name!r}; known: {known}") from None
