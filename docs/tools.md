# Tools

Helper scripts in [src/gem_mpc/tools/](../src/gem_mpc/tools/). Each is runnable as a
module and resolves its default input through
[paths.py](../src/gem_mpc/paths.py), so no working directory is assumed:

```bash
python -m gem_mpc.tools.plot_oscillation
python -m gem_mpc.tools.plot_oscillation /path/to/other.csv    # most accept an override
```

Anything that produces a figure writes it to `results/`.

## Driving

| Script | Purpose |
|---|---|
| `manual_driver.py` | Keyboard teleop publishing `AckermannDrive`. Use with the recorder to collect data by hand |
| `auto_driver.py` | Scripted speed and steering sweeps, for repeatable excitation |

Both need ROS and a running simulator.

## Data inspection

| Script | Reads | Purpose |
|---|---|---|
| `inspect_data.py` | `data/*.csv` | Per-file sample counts and the straight vs. turning split; writes `data_distribution.png` |
| `calc_max_rates.py` | `data/*.csv` | Maximum observed acceleration and steering rate, for sanity-checking the `MAX_ACCEL` / `MAX_STEER_RATE` limits. Differentiates over consecutive 1 ms rows, so its acceleration figures are noise-dominated; see the [caveat](mpc.md#constraints). Accepts `--data-dir` |
| `model_check.py` | `data/`, `models/` | Test-split RMSE **and bias** per output dimension. The in-distribution check on a trained model |

## Controller log analysis

All read `results/mpc_debug.csv` by default.

| Script | Purpose |
|---|---|
| `plot_oscillation.py` | Six-panel diagnostic: steering command vs. actual, CTE, speed, yaw rate (odom vs. IMU), IMU accelerations. Start here when something looks wrong |
| `analyze_cte.py` | Mean cross-track error split by left turn, right turn and straight. A left/right asymmetry points at a steering calibration offset |
| `analyze_log.py` | Finds the first step where abs(CTE) exceeds 1 m and prints the surrounding rows, for locating where a run started to diverge |
| `analyze_85s.py` | Dumps a fixed 84 to 87 s window row by row. Written for one specific incident; edit the window for another |
| `plot_trajectory.py` | Reads `results/mpc_trajectories.pkl` and plots the reference against the predicted horizon at a chosen time (`python -m gem_mpc.tools.plot_trajectory 85.0`) |

## Path and reference

All read `waypoints/wps.csv` by default.

| Script | Purpose |
|---|---|
| `plot_path.py` | High-resolution plot of the raw waypoint path |
| `analyze_path.py` | Segment-length statistics, for spotting gaps or duplicated waypoints |
| `check_yaw.py` | Heading continuity after downsampling; reports the largest wrapped jump |
| `test_spline.py` | Periodic B-spline fit against the raw waypoints, the same fit `mpc.py` uses |
| `plot_velocity_profile.py` | Curvature-derived speed profile and the curvature/velocity correlation |
| `compare_velocity.py` | The velocity column stored in `wps.csv` against the profile the controller computes |

`verify_wrap.py` is a standalone self-check of the arc-length wraparound logic used
for lookahead on a closed path. It takes no input and asserts its own expectations.

## Note on staleness

These scripts accumulated during debugging and are not covered by tests. `analyze_85s.py`
in particular hardcodes a time window from one investigation. The three `analyze_*`
scripts were written against an older `mpc_debug.csv` schema and have been updated to
the current column names (`cte_signed`, `steer_cmd_pub`); `analyze_85s.py` previously
printed per-wheel steering angles that the controller no longer logs, and those two
columns were dropped from its output.
