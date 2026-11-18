# Interview guide: flightsim-fault-harness

Eight questions an interviewer could ask about this repo, with answers, followed by
the things you should be able to say about the repo without looking.

## Questions and answers

### 1. Walk me through the altitude-hold loop. Why cascaded, and what did the gain sweep show?

Altitude error is turned into a commanded vertical speed, clamped to +400 / -500 fpm.
A PI loop on vertical-speed error, plus pitch-rate damping, produces a "nose up" demand
that is added to the trim elevator (sign flipped, because in JSBSim positive elevator is
trailing-edge down, nose down). It is cascaded because a direct altitude-to-elevator PID
on a 172 is hard to stabilise and would pitch hard at a distant target; the clamp on the
intermediate command turns a step in target into a smooth capture at a sane rate, and the
inner loop's integrator finds the trim elevator as speed and altitude change. The sweep
(`scripts/gain_sweep.py`) showed the inner loop tracked commanded vertical speed almost
exactly, so slow capture was the outer gain: 1 fpm per ft gives a 60 s time constant,
3 fpm per ft gives 20 s. Weak answer: describing it as "a PID on altitude".

### 2. Why does the climb limit differ from the descent limit?

Because at 100 kts between 5000 and 6000 ft the `c172x` model saturates the throttle at
500 fpm; the throttle loop pinned at 1.0 and the vertical loop fought an unreachable
command. 400 fpm keeps the throttle inside its authority. The README says explicitly that
whether this matches the real aircraft was not checked. Weak answer: quoting a POH climb
rate without noting that 100 kts is well above Vy.

### 3. The detectors never see truth. Where exactly is that boundary enforced?

`sensors.py` builds a frame with only seven channels (airspeed, altitude, vertical speed,
pitch, roll, heading, RPM) plus time; `detectors.py` and `triage/features.py` read that
frame only, and a test asserts the sensor frame carries no command or target columns.
Truth and `faults.json` are read by `evaluate.py` after the fact to score alarms, and by
`triage/evaluate.py` to label windows that were chosen from alarms alone. Weak answer:
"the detectors are in a different file".

### 4. How were detector thresholds chosen, and why is staleness treated differently?

Nominal runs are split by sorted id into validation, training and hold-out. Residual
scales and regression coefficients and the Isolation Forest are fitted on training runs.
For each statistical detector the threshold is the lowest quantile of the pooled
validation score at which alarm events (rising edges, with a 5-sample persistence) occur at
or below 1 per flight hour. Staleness is a rule, "channel unchanged for 3 s", with the
threshold fixed at 1.0: when it was tuned by the same search it fell to 1.4 s and fired on
chance streaks in quantised channels. False alarms are reported on hold-out runs only.
Weak answer: "I picked thresholds that looked right".

### 5. What does the confusion table tell you, and what would you do about attitude_noise_burst?

Each fault is caught by the detector you would expect from physics: altimeter faults by
`alt_rate`, frozen pitot and dropout by `staleness`, engine power loss by `airspeed` and
`energy`, control lag by all four residuals and the forest. Attitude noise bursts are
caught in 1 of 4 runs, by the forest only, because no residual check looks at attitude
noise and the window-std feature is not sensitive at a 1-per-hour threshold. A fix is a
dedicated residual (pitch/roll high-frequency variance vs its nominal level) or a
rate-consistency check between pitch and vertical speed. It was left as a documented gap
rather than tuned away. Weak answer: proposing to lower the forest threshold without
mentioning the false-alarm cost.

### 6. Why is `engine_power_loss` modelled as throttle authority, and what is the risk?

JSBSim's piston engine exposes running/not-running and the throttle input, no
partial-power knob, so the injector scales the throttle reaching the engine by
`1 - magnitude`. The risk is that the autopilot's throttle integrator partly compensates,
and a real partial power loss changes RPM and roughness differently; both are in the
README. Weak answer: claiming it edits an engine property.

### 7. What is the replay gate and why does it exist?

`tests/test_replay.py` replays five recorded faulted runs stored under `runs/fixtures/`
(sensors and fault log only, no truth) through the committed model and asserts each is
detected within `max_ttd_s` with at most the recorded number of pre-onset alarms. Any
change to detectors, thresholds or the fitted model must keep it passing, and CI runs it
on every push. It exists because a threshold or feature change can silently trade
detection for false alarms; the campaign evaluation measures that in aggregate, the gate
pins specific known-good cases. Weak answer: "it is just more tests".

### 8. How is the triage agent evaluated without leaking ground truth, and what does calibration mean here?

Windows are 15 s before to 30 s after each alarm onset, plus two fixed level-flight
windows per nominal run; the fault log is used only afterwards to label each window.
The agent gets the alarm list and per-channel statistics, returns a pydantic-validated
report, and every response is cached by hash so re-evaluation is offline. Accuracy is per
fault type on fault windows; the "unknown" rate on nominal windows measures whether it
invents faults; calibration bins confidence and compares it with observed accuracy,
summarised as expected calibration error. The accuracy is conditional on detection: a
missed fault never reaches the agent. Weak answer: not mentioning the cache or the
conditional nature of the score.

## What you should know about this repo cold

- **Purpose.** A simulator test bench: inject faults into a flown Cessna 172, detect them
  from sensor channels only, triage them with an LLM, score everything against injected
  ground truth, and gate releases on replaying recorded faults.
- **Pipeline.** scenario -> autopilot -> JSBSim (120 Hz) -> truth (10 Hz) -> sensors ->
  faults (sensor or aircraft) -> detectors -> alarms -> triage agent -> evaluation -> results.
- **The three commands.** `make campaign fit`, `make test results`, optionally `make triage`.
- **Determinism.** JSBSim and the autopilot are deterministic; the seed drives sensor
  noise and the campaign's fault draws. Same seed, same bytes.
- **Numbers are generated.** Every table in the README comes from `make results`; the
  per-run table is `results/runs.md`, the LLM section reads `triage_cache/` offline.
- **Known gaps.** No rudder loop; attitude noise mostly missed; false-alarm rate from
  1.1 held-out hours; sensor noise levels are guesses; one aircraft, one profile.
- **Where to add things.** A fault: `faults.py` + `campaign.py` ranges + `triage/schema.py`
  literal + a test + a fixture. A detector: a score column in `detectors.py`, add its name
  to `DETECTORS`, refit, check the replay gate.
