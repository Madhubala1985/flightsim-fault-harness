"""A deliberately simple three-axis autopilot for the Cessna 172.

Three independent loops, each built from the same ``PID`` block:

* **Altitude hold (elevator).** Cascaded. The outer loop turns altitude error
  into a commanded vertical speed, clamped so the aircraft climbs or descends
  at a sane rate instead of pitching hard at a far-away target. The inner loop
  is a PI on vertical-speed error, plus pitch-rate damping, driving the
  elevator. The integrator is what finds the trim elevator position.
* **Heading hold (aileron).** Cascaded. The outer loop turns heading error
  (wrapped to +/-180 deg) into a commanded bank angle, clamped to a gentle
  bank. The inner loop is a PI on bank error, plus roll-rate damping, driving
  the aileron. The integrator cancels the steady rolling moment from engine
  torque and propeller effects.
* **Airspeed hold (throttle).** A single PI on airspeed error driving throttle.

Sign conventions were checked empirically against the bundled ``c172x`` model
from a trimmed state (see ``tests/test_flight.py::test_control_sign_conventions``):
positive elevator command pitches the nose DOWN (trailing edge down), positive
aileron command rolls right, positive throttle raises RPM. ``ELEVATOR_NOSE_UP``
carries the elevator sign so the loop below can be written in "nose up" terms.
Gains were hand-tuned on the ``basic`` scenario; they are not claimed to be
optimal, only stable and easy to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FPM_PER_FPS = 60.0
ELEVATOR_NOSE_UP = -1.0  # multiply a "nose up" demand by this to get an elevator command


def wrap_deg(angle: float) -> float:
    """Wrap an angle in degrees to the range [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp ``x`` to the closed interval [lo, hi]."""
    return lo if x < lo else hi if x > hi else x


@dataclass
class PID:
    """Textbook PID with output clamping and clamp-based anti-windup.

    ``update`` returns ``kp*e + ki*integral(e) + kd*de/dt``. If the clamped
    output is saturated and the error pushes further into saturation, the
    integrator is frozen so it cannot wind up.
    """

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    out_min: float = -1.0
    out_max: float = 1.0
    integral: float = field(default=0.0, init=False)
    prev_error: float | None = field(default=None, init=False)

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = None

    def update(self, error: float, dt: float) -> float:
        derivative = 0.0
        if self.prev_error is not None and dt > 0:
            derivative = (error - self.prev_error) / dt
        self.prev_error = error

        candidate = self.integral + error * dt
        unclamped = self.kp * error + self.ki * candidate + self.kd * derivative
        output = clamp(unclamped, self.out_min, self.out_max)
        if output == unclamped or (unclamped > self.out_max) != (error > 0):
            self.integral = candidate
        return output


@dataclass
class AutopilotGains:
    """All tunable numbers in one place so they can be explained and tested."""

    # Altitude outer loop: fpm of commanded vertical speed per ft of error.
    # 3 fpm/ft means the last part of a capture decays with a ~20 s time constant.
    alt_to_vs: float = 3.0
    # A C172 at 100 kts cannot sustain 500 fpm at 5-6000 ft without saturating
    # the throttle, so the climb limit is lower than the descent limit.
    vs_climb_max_fpm: float = 400.0
    vs_descent_max_fpm: float = 500.0
    # Vertical-speed inner loop (elevator per fps of error) and pitch-rate damping.
    vs_kp: float = 0.06
    vs_ki: float = 0.01
    pitch_rate_damping: float = 2.0
    # Heading outer loop: deg of bank per deg of heading error.
    hdg_to_bank: float = 1.0
    bank_max_deg: float = 20.0
    # Bank inner loop (aileron per deg of error) and roll-rate damping.
    bank_kp: float = 0.01
    bank_ki: float = 0.003
    roll_rate_damping: float = 0.2
    # Airspeed loop: throttle per kt of error.
    ias_kp: float = 0.03
    ias_ki: float = 0.002


@dataclass(frozen=True)
class Trim:
    """Feed-forward control positions for level flight, found by JSBSim's trim.

    The autopilot only adds corrections around these, so its integrators start
    near zero and do not have to discover the trim state from scratch.
    """

    elevator: float = 0.0
    aileron: float = 0.0
    throttle: float = 0.0


@dataclass
class Targets:
    """What the autopilot is asked to fly."""

    altitude_ft: float
    heading_deg: float
    airspeed_kts: float


@dataclass
class State:
    """The subset of aircraft state the autopilot reads (truth, in aviation units)."""

    altitude_ft: float
    vertical_speed_fpm: float
    pitch_rate_rad_s: float
    roll_deg: float
    roll_rate_rad_s: float
    heading_deg: float
    airspeed_kts: float


@dataclass
class Commands:
    """Normalised control surface commands in [-1, 1] (throttle in [0, 1])."""

    elevator: float
    aileron: float
    throttle: float
    # Intermediate commands, logged so the behaviour can be explained offline.
    vs_cmd_fpm: float
    bank_cmd_deg: float


class Autopilot:
    """Stateful controller: call ``step`` once per simulation step."""

    def __init__(self, gains: AutopilotGains | None = None, trim: Trim | None = None) -> None:
        self.gains = gains or AutopilotGains()
        self.trim = trim or Trim()
        g = self.gains
        self.vs_pid = PID(kp=g.vs_kp, ki=g.vs_ki, out_min=-1.0, out_max=1.0)
        self.bank_pid = PID(kp=g.bank_kp, ki=g.bank_ki, out_min=-1.0, out_max=1.0)
        self.ias_pid = PID(kp=g.ias_kp, ki=g.ias_ki, out_min=-1.0, out_max=1.0)

    def step(self, state: State, targets: Targets, dt: float) -> Commands:
        g = self.gains

        # --- altitude -> vertical speed -> elevator ---
        alt_err = targets.altitude_ft - state.altitude_ft
        vs_cmd = clamp(g.alt_to_vs * alt_err, -g.vs_descent_max_fpm, g.vs_climb_max_fpm)
        vs_err_fps = (vs_cmd - state.vertical_speed_fpm) / FPM_PER_FPS
        nose_up = self.vs_pid.update(vs_err_fps, dt) - g.pitch_rate_damping * state.pitch_rate_rad_s
        elevator = self.trim.elevator + ELEVATOR_NOSE_UP * nose_up

        # --- heading -> bank -> aileron ---
        hdg_err = wrap_deg(targets.heading_deg - state.heading_deg)
        bank_cmd = clamp(g.hdg_to_bank * hdg_err, -g.bank_max_deg, g.bank_max_deg)
        bank_err = bank_cmd - state.roll_deg
        aileron = (
            self.trim.aileron
            + self.bank_pid.update(bank_err, dt)
            - g.roll_rate_damping * state.roll_rate_rad_s
        )

        # --- airspeed -> throttle ---
        ias_err = targets.airspeed_kts - state.airspeed_kts
        throttle = self.trim.throttle + self.ias_pid.update(ias_err, dt)

        return Commands(
            elevator=clamp(elevator, -1.0, 1.0),
            aileron=clamp(aileron, -1.0, 1.0),
            throttle=clamp(throttle, 0.0, 1.0),
            vs_cmd_fpm=vs_cmd,
            bank_cmd_deg=bank_cmd,
        )
