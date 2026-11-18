### Triage agent (ollama / llama3.2:3b, 261 windows, served from cache: 261, new calls: 0)

The agent sees only the alarm list and a sensor feature summary for each window, never truth or the fault log. Windows are chosen from alarms alone; the label is assigned afterwards from the fault log.

Overall accuracy on fault windows: 0.24. Faulted runs with no alarm (untriaged): 7.

| fault_type | windows | accuracy | said_unknown | mean_confidence |
|---|---|---|---|---|
| altimeter_bias | 19 | 0.00 | 1.00 | 0.00 |
| altimeter_drift | 21 | 0.00 | 1.00 | 0.00 |
| attitude_noise_burst | 3 | 0.00 | 1.00 | 0.00 |
| control_lag | 14 | 0.00 | 0.29 | 0.58 |
| engine_power_loss | 19 | 0.95 | 0.05 | 0.80 |
| sensor_dropout | 20 | 0.40 | 0.55 | 0.34 |
| stuck_airspeed | 14 | 0.00 | 0.57 | 0.38 |

Nominal windows: 151; said 'unknown' on 0.84 of them, named a fault on 0.16.

Calibration (expected calibration error 0.506):

| confidence_bin | windows | mean_confidence | accuracy |
|---|---|---|---|
| [0.0, 0.5) | 2 | 0.00 | 1.00 |
| [0.7, 0.9) | 57 | 0.84 | 0.32 |
| [0.9, 1.0) | 8 | 1.00 | 0.75 |
