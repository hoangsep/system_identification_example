# Neural MPC

[src/gem_mpc/mpc.py](../src/gem_mpc/mpc.py) is a ROS node that solves a nonlinear OCP
every 50 ms with [acados](https://github.com/acados/acados), using the network from
[system identification](system_identification.md) as the prediction model.

```bash
python -m gem_mpc.mpc        # or: gem-mpc
```

## How the network gets into the solver

The torch weights are loaded once, then rebuilt as a symbolic CasADi expression:
`ca.mtimes` for each layer, `ca.tanh` between them, with the scaler mean/scale folded
in as `ca.DM` constants. The result becomes `model.disc_dyn_expr`, so acados
generates C code that evaluates and differentiates the network directly. There is no
torch call in the control loop.

Normalisation is applied inside the expression, so the solver works in physical units
while the network sees normalised ones.

## Formulation

**State** `x` (8):

| Index | Symbol | Meaning |
|---|---|---|
| 0, 1 | `X`, `Y` | global position, m |
| 2 | `yaw` | heading, unwrapped, rad |
| 3 | `v` | speed, m/s |
| 4 | `steer` | actual steering angle, rad |
| 5 | `yaw_rate` | rad/s |
| 6 | `prev_cmd_steer` | previous commanded steering, rad |
| 7 | `prev_cmd_v` | previous commanded speed, m/s |

**Control** `u` (2): `cmd_v`, `cmd_steer`.

**Parameters** `p` (4): the per-node reference `ref_x`, `ref_y`, `ref_yaw`, `ref_v`.

Carrying the previous command in the state is what makes rate limits expressible: the
solver can constrain `u - x[6:8]` as an ordinary linear constraint, and the network
gets the `prev_cmd_*` inputs it was trained with. The dynamics shift them each step
(`prev_cmd_steer <- cmd_steer`).

Yaw is kept unwrapped in the state so the horizon stays continuous across the
`+/-pi` boundary; heading error in the cost is wrapped explicitly with
`atan2(sin(d), cos(d))`.

## Cost

`NONLINEAR_LS` on the residual vector below. Position error enters as a squared
distance, so its residual is already `e_pos` and the weight acts on the fourth power
of distance; this is intentional, it makes large excursions dominate.

| Residual | Weight | Value |
|---|---|---|
| `(X - ref_x)^2 + (Y - ref_y)^2` | `Q_lat` | 1.0 |
| wrapped `yaw - ref_yaw` | `Q_yaw` | 1.0 |
| `v - ref_v` | `Q_vel` | 5.0 |
| `cmd_v - v` | `R_track_v` | 1.0 |
| `cmd_steer` | `R_steer` | 20.0 |
| `steer` | (unused) | 0.0 |
| `cmd_steer - prev_cmd_steer` | `R_steer_rate` | 2000.0 |
| `cmd_v - prev_cmd_v` | `R_cmdv_rate` | 50.0 |

Terminal cost keeps the first three terms only.

`R_steer_rate` at 2000 is two orders of magnitude above everything else. Steering
chatter was the dominant failure mode: the model's lateral response is strong enough
that the optimizer will happily oscillate the wheel between steps unless the rate is
expensive. `results/oscillation_debug.png` is the plot for diagnosing this.

## Constraints

State bounds (`idxbx = [3, 4]`):

| Quantity | Range |
|---|---|
| `v` | `[0, MAX_REF_V]` = `[0, 5.5]` m/s |
| `steer` | `[-0.65, 0.65]` rad |

Control bounds (`idxbu = [0, 1]`): `cmd_v` in `[0, 5.5]` m/s, `cmd_steer` in
`[-0.6, 0.6]` rad.

Rate limits, as general linear constraints coupling `u` to the previous command in
the state:

| Constraint | Limit | Per step at `DT = 0.05` |
|---|---|---|
| `cmd_steer - prev_cmd_steer` | `MAX_STEER_RATE = 0.5` rad/s | 0.025 rad |
| `cmd_v - prev_cmd_v` | `MAX_ACCEL = 2.0` m/s² | 0.1 m/s |

These are hard limits on top of the soft rate penalties in the cost.

`tools/calc_max_rates.py` reports the rates observed in the recorded logs and is the
tool for revisiting these numbers, but read its output with care: it differentiates
`v_actual` between consecutive raw rows, which are 1 ms apart, so quantisation noise
dominates and it reports peaks above 60 m/s² on the 1 kHz logs. The smoother
moderate-speed logs (`gem_data_a1`, `gem_data_a2`) give the plausible envelope, around
2.4 m/s² and 0.6 rad/s, which is the range the constants above sit in.

## Solver

| Option | Value |
|---|---|
| `integrator_type` | `DISCRETE` (the network is already a 1-step map) |
| `nlp_solver_type` | `SQP_RTI` |
| `qp_solver` | `PARTIAL_CONDENSING_HPIPM` |
| `N_horizon` | `HORIZON` = 20 |
| `tf` | 1.0 s |

SQP-RTI performs one QP per control step rather than iterating to convergence, which
is the standard real-time compromise: the solution improves across steps as the
problem shifts. Generated C code and `acados_ocp.json` go to `build/acados/`.

Each solve is warm-started from the previous solution shifted by one step
(`build_initial_guess`). On the first step, or when the previous solve failed, it
falls back to a forward rollout of the model under the reference.

## Reference generation

1. `waypoints/wps.csv` is downsampled to a 0.1 m minimum spacing and fitted with a
   periodic B-spline (`scipy.interpolate.splprep`), giving a smooth closed path with
   analytic derivatives.
2. Curvature from the spline sets a speed profile, clamped to
   `[MIN_REF_V, MAX_REF_V]` = `[0.5, 5.5]` m/s.
3. Each control step, the closest path index is found and the horizon reference is
   sampled ahead by arc length, so reference spacing follows the speed rather than a
   fixed index stride.
4. The target speed is ramped at `VREF_RAMP_ACCEL` = 1.0 m/s² and allowed at most
   `VADV_EXTRA` = 0.3 m/s above the measured speed, which stops the reference running
   away from a vehicle that cannot keep up.

`tools/plot_path.py`, `tools/test_spline.py` and `tools/plot_velocity_profile.py`
inspect stages 1 to 3 without running the controller.

## Latency compensation and smoothing

`compensate_latency` rolls the measured state forward by one step through the model
before solving, so the solution applies to the state the vehicle will be in when the
command lands rather than the one it has already left.

Published commands are then low-pass filtered: `ALPHA_STEER` = 0.90, `ALPHA_V` = 0.80
(higher means more weight on the new command). `STEER_GAIN_COMP` and `STEER_OFFSET`
are available for steering calibration and are currently 1.0 and 0.0.

## ROS interface

| Direction | Topic | Type |
|---|---|---|
| subscribe | `/gazebo/model_states` | `gazebo_msgs/ModelStates` |
| subscribe | `/gem/joint_states` | `sensor_msgs/JointState` |
| subscribe | `/gem/imu` | `sensor_msgs/Imu` |
| publish | `/gem/ackermann_cmd` | `ackermann_msgs/AckermannDrive` |
| publish | `/gem/mpc_debug` | `visualization_msgs/MarkerArray` (latched) |

Add `/gem/mpc_debug` as a MarkerArray display in RViz to see the reference and the
predicted horizon live.

## Online model evaluation

Alongside control, the node compares each step's predicted delta against what
actually happened, accepting a sample only when the measured `dt` is within
`EVAL_DT_TOL` = 0.02 of `DT`. The running per-dimension RMSE is written to
`results/mpc_rmse.png` on shutdown. This is a closed-loop check on the same model the
offline validation scores, and it is the one that reveals distribution shift once the
controller starts visiting states the training data never covered.

## Outputs

Written on shutdown to `results/`:

| File | Contents |
|---|---|
| `mpc_debug.csv` | Per-step log, 20 Hz ([columns](results.md#mpc_debugcsv-columns)) |
| `mpc_trajectories.pkl` | Reference and predicted horizon at every step |
| `mpc_cte_signed.png` | Signed cross-track error over time |
| `mpc_rmse.png` | Running 1-step model RMSE per output dimension |

## Tuning reference

| Constant | Value | Effect |
|---|---|---|
| `HORIZON` | 20 | Lookahead in steps; longer costs solve time |
| `DT` | 0.05 | Control period; must match the training `TARGET_DT` |
| `MIN_REF_V`, `MAX_REF_V` | 0.5, 5.5 | Speed profile clamp, m/s |
| `MAX_ACCEL` | 2.0 | Hard command acceleration limit, m/s² |
| `MAX_STEER_RATE` | 0.5 | Hard command steering rate limit, rad/s |
| `ALPHA_STEER`, `ALPHA_V` | 0.90, 0.80 | Command smoothing |
| `VREF_RAMP_ACCEL` | 1.0 | Reference speed ramp, m/s² |
| `VADV_EXTRA` | 0.3 | Reference speed allowance over measured, m/s |
| `Q_lat`, `Q_yaw`, `Q_vel` | 1, 1, 5 | Tracking weights |
| `R_steer`, `R_steer_rate` | 20, 2000 | Steering effort and chatter penalties |

`DT` is the one value that cannot be changed on its own: it is an input feature of the
network, and the training data is filtered to `0.05 ± 0.01`. Changing it means
retraining.
