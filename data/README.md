# Recorded driving data

CSV logs from the Polaris GEM Gazebo simulator, written by
[data_recorder.py](../src/gem_mpc/data_recorder.py) and consumed by
[train_model.py](../src/gem_mpc/train_model.py), which globs `*.csv` in this
directory.

## Schema

One row per `/gazebo/model_states` message, so the raw rate is about 1 kHz. Training
pairs each row with the one 0.05 s later (the control period) rather than decimating,
so all rows contribute; see
[system identification](../docs/system_identification.md#preprocessing).

| Column | Unit | Source | Meaning |
|---|---|---|---|
| `time` | s | `rospy.get_time()` | ROS time at the sample |
| `cmd_speed` | m/s | `/gem/ackermann_cmd` | Last commanded speed |
| `cmd_steer` | rad | `/gem/ackermann_cmd` | Last commanded steering angle |
| `steer_actual` | rad | `/gem/joint_states` | Mean of the left and right steering hinge positions |
| `steer_rate` | rad/s | `/gem/joint_states` | Mean of the left and right steering hinge velocities |
| `x`, `y` | m | `/gazebo/model_states` | Global position |
| `yaw` | rad | `/gazebo/model_states` | Heading from the pose quaternion |
| `v_actual` | m/s | `/gazebo/model_states` | Scalar speed, `sqrt(vx² + vy²)` |
| `yaw_rate` | rad/s | `/gazebo/model_states` | Angular velocity about z |

`train_model.py` requires `time`, `cmd_speed`, `cmd_steer`, `steer_actual`, `x`, `y`,
`yaw`, `v_actual` and `yaw_rate`. `steer_rate` is recorded but not currently used as a
training input; `tools/calc_max_rates.py` uses it when present.

Steering commands are positive left, matching the ROS convention.

## Bundled logs

| File | Rows | Duration | Max speed | Max abs steer | Character |
|---|---|---|---|---|---|
| `gem_data.csv` | 425,777 | 426 s | 5.50 m/s | 0.61 rad | Long mixed session at the full speed range |
| `gem_data_a2.csv` | 199,606 | 200 s | 3.92 m/s | 0.50 rad | Moderate speed, sustained cornering |
| `gem_data_a1.csv` | 170,822 | 171 s | 3.93 m/s | 0.50 rad | Moderate speed, sustained cornering |
| `acceleration_2.csv` | 54,281 | 54 s | 5.51 m/s | 0.61 rad | Acceleration and braking transients |
| `accelerate_1.csv` | 47,784 | 48 s | 5.58 m/s | 0.61 rad | Acceleration and braking transients |

898,270 raw rows total, yielding 897,470 training samples. The two `accelerate*` logs exist specifically to cover
longitudinal transients, which are underrepresented in steady-speed driving and are
where the model's `d_v` error concentrates
([results](../docs/results.md#online-model-accuracy)).

## Recording more

```bash
# terminal 1: simulator running, then
python -m gem_mpc.data_recorder

# terminal 2: drive
python -m gem_mpc.tools.manual_driver
```

The recorder writes `data/gem_data_<timestamp>.csv` on shutdown, so new runs are
picked up by training automatically and never overwrite an existing log. Use `--out`
for a specific path.

Aim for coverage rather than duration: `python -m gem_mpc.tools.inspect_data` reports
the straight vs. turning balance. The bundled logs sit at 32.5% straight and 67.5%
turning, which is a reasonable target; unguided cruising drifts well below it.

## A note on size

These logs total about 150 MB and are tracked in git, which makes cloning slow. They
are kept in-tree so training reproduces without a separate download. If you fork this
for your own work, consider moving them to release assets or Git LFS.
