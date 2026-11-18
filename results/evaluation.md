### Detection per fault type (102 faulted runs, any detector)

| fault_type | runs | detection_rate | median_ttd_s | pre_onset_false_alarms | pre_onset_per_hour |
|---|---|---|---|---|---|
| altimeter_bias | 12 | 1.00 | 0.50 | 2 | 2.26 |
| altimeter_drift | 13 | 1.00 | 5.80 | 0 | 0.00 |
| attitude_noise_burst | 15 | 0.07 | 118.60 | 2 | 1.47 |
| control_lag | 11 | 1.00 | 8.80 | 0 | 0.00 |
| engine_power_loss | 19 | 1.00 | 4.10 | 1 | 0.57 |
| sensor_dropout | 20 | 1.00 | 3.30 | 1 | 0.66 |
| stuck_airspeed | 12 | 1.00 | 3.50 | 1 | 0.82 |

Time-to-detect is measured from fault onset; a run counts as detected if an alarm starts between onset and 10 s after the fault ends.

### False alarms on held-out nominal runs (14 runs, 2.57 flight hours)

| detector | events | per_hour |
|---|---|---|
| alt_rate | 0 | 0.00 |
| airspeed | 5 | 1.95 |
| energy | 0 | 0.00 |
| staleness | 0 | 0.00 |
| iforest | 0 | 0.00 |
| any | 5 | 1.95 |

Held-out = nominal runs used neither for training nor for threshold validation.

Pooled estimate ('any' detector) over all fault-free time — 2.57 held-out nominal hours plus 8.37 pre-onset hours on faulted runs: 12 alarm events in 10.94 hours, about 1.1 per hour.
