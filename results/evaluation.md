### Detection per fault type (41 faulted runs, any detector)

| fault_type | runs | detection_rate | median_ttd_s | pre_onset_false_alarms |
|---|---|---|---|---|
| altimeter_bias | 3 | 1.00 | 0.50 | 1 |
| altimeter_drift | 6 | 1.00 | 8.65 | 1 |
| attitude_noise_burst | 4 | 0.25 | 118.60 | 1 |
| control_lag | 6 | 1.00 | 11.45 | 1 |
| engine_power_loss | 7 | 1.00 | 3.60 | 0 |
| sensor_dropout | 8 | 1.00 | 3.30 | 1 |
| stuck_airspeed | 7 | 1.00 | 3.50 | 3 |

Time-to-detect is measured from fault onset; a run counts as detected if an alarm starts between onset and 10 s after the fault ends.

### False alarms on held-out nominal runs (6 runs, 1.10 flight hours)

| detector | events | per_hour |
|---|---|---|
| alt_rate | 0 | 0.00 |
| airspeed | 0 | 0.00 |
| energy | 0 | 0.00 |
| staleness | 0 | 0.00 |
| iforest | 0 | 0.00 |
| any | 0 | 0.00 |

Held-out = nominal runs used neither for training nor for threshold validation.
