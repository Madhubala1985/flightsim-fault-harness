"""Run a scripted flight in JSBSim under the autopilot and log truth at 10 Hz.

The sim itself is deterministic: JSBSim has no randomness in this
configuration and the autopilot is pure arithmetic. The seed is recorded in the
run metadata and is consumed by ``fsfh.sensors`` to generate sensor noise, so
"same seed" means "same truth and same sensor channels".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jsbsim
import numpy as np
import pandas as pd

from fsfh.autopilot import FPM_PER_FPS, Autopilot, AutopilotGains, State, Trim
from fsfh.faults import AircraftFaultInjector, Fault
from fsfh.scenarios import Scenario

LOG_HZ = 10
MIXTURE_CMD = 0.85  # fixed; the autopilot does not manage mixture

# Every JSBSim property read or written, checked against
# fdm.query_property_catalog() in tests/test_flight.py::test_property_names_exist.
PROPERTIES = {
    "altitude_ft": "position/h-sl-ft",
    "ias_kts": "velocities/vc-kts",
    "vs_fps": "velocities/h-dot-fps",
    "roll_rad": "attitude/phi-rad",
    "pitch_rad": "attitude/theta-rad",
    "heading_rad": "attitude/psi-rad",
    "p_rad_s": "velocities/p-rad_sec",
    "q_rad_s": "velocities/q-rad_sec",
    "r_rad_s": "velocities/r-rad_sec",
    "rpm": "propulsion/engine/engine-rpm",
    "alpha_deg": "aero/alpha-deg",
    "nz_g": "accelerations/Nz",
    "lat_deg": "position/lat-gc-deg",
    "lon_deg": "position/long-gc-deg",
    "elevator_cmd": "fcs/elevator-cmd-norm",
    "aileron_cmd": "fcs/aileron-cmd-norm",
    "throttle_cmd": "fcs/throttle-cmd-norm",
    "mixture_cmd": "fcs/mixture-cmd-norm",
    "do_simple_trim": "simulation/do_simple_trim",
}

TRUTH_COLUMNS = [
    "t_s",
    "altitude_ft",
    "ias_kts",
    "vs_fpm",
    "roll_deg",
    "pitch_deg",
    "heading_deg",
    "p_rad_s",
    "q_rad_s",
    "r_rad_s",
    "rpm",
    "alpha_deg",
    "nz_g",
    "lat_deg",
    "lon_deg",
    "elevator_cmd",
    "aileron_cmd",
    "throttle_cmd",
    "alt_cmd_ft",
    "heading_cmd_deg",
    "ias_cmd_kts",
    "vs_cmd_fpm",
    "bank_cmd_deg",
]


@dataclass
class FlightResult:
    truth: pd.DataFrame
    dt_s: float
    jsbsim_version: str
    trim: Trim


def make_fdm(scenario: Scenario) -> tuple[jsbsim.FGFDMExec, Trim]:
    """Create a JSBSim instance trimmed for level flight at the scenario's start.

    JSBSim's simple trim adjusts the pitch trim tab, aileron, rudder and
    throttle until forces and moments balance. Those positions are returned as
    the autopilot's feed-forward ``Trim``.
    """
    fdm = jsbsim.FGFDMExec(jsbsim.get_default_root_dir())
    fdm.set_debug_level(0)
    if not fdm.load_model(scenario.aircraft):
        raise RuntimeError(f"JSBSim could not load aircraft {scenario.aircraft!r}")
    fdm["ic/h-sl-ft"] = scenario.initial_altitude_ft
    fdm["ic/vc-kts"] = scenario.initial_airspeed_kts
    fdm["ic/gamma-deg"] = 0.0
    fdm["ic/psi-true-deg"] = scenario.initial_heading_deg
    fdm.run_ic()
    fdm["propulsion/set-running"] = -1  # start all engines
    fdm[PROPERTIES["mixture_cmd"]] = MIXTURE_CMD
    fdm[PROPERTIES["do_simple_trim"]] = 1
    trim = Trim(
        elevator=fdm[PROPERTIES["elevator_cmd"]],
        aileron=fdm[PROPERTIES["aileron_cmd"]],
        throttle=fdm[PROPERTIES["throttle_cmd"]],
    )
    return fdm, trim


def read_state(fdm: jsbsim.FGFDMExec) -> State:
    p = PROPERTIES
    return State(
        altitude_ft=fdm[p["altitude_ft"]],
        vertical_speed_fpm=fdm[p["vs_fps"]] * FPM_PER_FPS,
        pitch_rate_rad_s=fdm[p["q_rad_s"]],
        roll_deg=math.degrees(fdm[p["roll_rad"]]),
        roll_rate_rad_s=fdm[p["p_rad_s"]],
        heading_deg=math.degrees(fdm[p["heading_rad"]]) % 360.0,
        airspeed_kts=fdm[p["ias_kts"]],
    )


def fly(
    scenario: Scenario,
    gains: AutopilotGains | None = None,
    faults: list[Fault] | None = None,
) -> FlightResult:
    """Fly the scenario start to finish and return the 10 Hz truth log.

    Aircraft-level faults in ``faults`` are injected between the autopilot and
    JSBSim; sensor-level faults are ignored here (see ``fsfh.faults``).
    """
    fdm, trim = make_fdm(scenario)
    dt = fdm.get_delta_t()
    steps_per_log = round(1.0 / (LOG_HZ * dt))
    if not math.isclose(steps_per_log * dt, 1.0 / LOG_HZ, rel_tol=1e-6):
        raise RuntimeError(f"log rate {LOG_HZ} Hz is not a multiple of sim dt {dt}")

    autopilot = Autopilot(gains, trim)
    injector = AircraftFaultInjector(faults or [], dt)
    p = PROPERTIES
    rows: list[list[float]] = []
    # Step count, not float sim time, decides when to stop, so a 20 s run logs
    # exactly 200 samples (t = 0.0 ... 19.9) rather than 200 or 201 by rounding.
    n_steps = round(scenario.duration_s / dt)
    for step in range(n_steps):
        t = fdm.get_sim_time()
        targets = scenario.targets_at(t)
        state = read_state(fdm)
        cmd = autopilot.step(state, targets, dt)
        fdm[p["elevator_cmd"]] = injector.elevator(t, cmd.elevator)
        fdm[p["aileron_cmd"]] = cmd.aileron
        fdm[p["throttle_cmd"]] = injector.throttle(t, cmd.throttle)

        if step % steps_per_log == 0:
            rows.append(
                [
                    t,
                    state.altitude_ft,
                    state.airspeed_kts,
                    state.vertical_speed_fpm,
                    state.roll_deg,
                    math.degrees(fdm[p["pitch_rad"]]),
                    state.heading_deg,
                    state.roll_rate_rad_s,
                    state.pitch_rate_rad_s,
                    fdm[p["r_rad_s"]],
                    fdm[p["rpm"]],
                    fdm[p["alpha_deg"]],
                    fdm[p["nz_g"]],
                    fdm[p["lat_deg"]],
                    fdm[p["lon_deg"]],
                    cmd.elevator,
                    cmd.aileron,
                    cmd.throttle,
                    targets.altitude_ft,
                    targets.heading_deg,
                    targets.airspeed_kts,
                    cmd.vs_cmd_fpm,
                    cmd.bank_cmd_deg,
                ]
            )
        fdm.run()

    truth = pd.DataFrame(np.array(rows, dtype=np.float64), columns=TRUTH_COLUMNS)
    return FlightResult(truth=truth, dt_s=dt, jsbsim_version=jsbsim.__version__, trim=trim)
