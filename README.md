# Polaris GEM System Identification & Neural MPC (acados)

Python + ROS/Gazebo workflow to:
1) record Polaris GEM simulator data to CSV,
2) learn a neural 1‑step dynamics model (local-frame deltas),
3) run NMPC with **acados (SQP‑RTI)** using the learned model.

Key scripts:
- `data_recorder.py`: logs ROS topics to CSV
- `train_model.py`: trains the neural dynamics + saves scalers/plots
- `mpc.py`: neural MPC controller (acados + CasADi)
- Plot helpers: `plot_oscillation.py`, `plot_trajectory.py`

Pre-trained artifacts are included: `gem_dynamics.pth`, `gem_scaler.pkl`, `gem_scaler_arrays.npz`.

## Build & Run the Simulation
- Build the ROS/Gazebo image (uses `Dockerfile`):  
  ```bash
  docker compose build
  ```
- Start a container with GUI passthrough (needs X server on host):  
  ```bash
  xhost +local:root             # allow X11 (optional if already set)
  docker compose up -d
  docker exec -it gem_mpc_container bash
  ```
- Inside the container:  
  ```bash
  source /opt/ros/noetic/setup.bash
  cd /root/catkin_ws
  catkin_make                     # rebuild after mounting this repo into src/assignment
  source devel/setup.bash
  ```
- Launch Gazebo/RViz for the GEM simulator:  
  ```bash
  roslaunch gem_gazebo gem_gazebo_rviz.launch
  ```
  (Adjust the launch file as needed for your scenario/track.)
- In another shell in the same container, run the controller from this repo:  
  ```bash
  source /root/catkin_ws/devel/setup.bash
  cd /root/catkin_ws/src/assignment
  python mpc.py
  ```
  The node subscribes to `/gazebo/model_states`, `/gem/joint_states`, `/gem/imu` and publishes `/gem/ackermann_cmd` (and RViz debug markers on `/gem/mpc_debug`).

## System Identification Workflow
- **Data capture**: Run the simulator, then log data with `python data_recorder.py` (from this repo with ROS environment sourced). Drive with `manual_driver.py` (keyboard) or `auto_driver.py` to excite the system.
  - By default the recorder writes `gem_data.csv` in the repo root; move/rename it into `neo_data/` (e.g. `mv gem_data.csv neo_data/run_001.csv`) or pass a different directory to training via `--data-dir`.
  - Required columns for training: `time`, `cmd_speed`, `cmd_steer`, `steer_actual`, `x`, `y`, `yaw`, `v_actual`, `yaw_rate` (`steer_rate` is optional).
- **Training**: Fit the neural dynamics model and scalers (default reads `neo_data/*.csv`):  
  ```bash
  python train_model.py
  ```
  Outputs:
  - `gem_dynamics.pth`
  - `gem_scaler.pkl`, `gem_scaler_arrays.npz`
  - `sysid_validation_plot.png` (actual vs. predicted deltas + scatter)
  - `sysid_input_output.png` (input/output overview)
  - `rmse_plot.png` (RMSE by subset)

## MPC Controller
- `mpc.py` reconstructs the trained network inside CasADi and solves NMPC with **acados (SQP‑RTI)** over a horizon of size `HORIZON` at timestep `DT`.
- The controller uses the learned 1‑step deltas `(dx_local, dy_local, d_yaw, d_v, d_steer)`, warm-starts from the previous solution, enforces accel/steer-rate constraints, and publishes `AckermannDrive` commands.
- On shutdown it writes `mpc_debug.csv`, `mpc_trajectories.pkl`, and saves plots `mpc_cte_signed.png` and `mpc_rmse.png`. Use `python plot_oscillation.py mpc_debug.csv` to generate `oscillation_debug.png`.

## Dockerfile (Dependencies)
- Base: `osrf/ros:noetic-desktop-full`.
- Apt: `ros-noetic-ackermann-msgs`, `ros-noetic-ros-control`, `ros-noetic-gazebo-ros-control`, `ros-noetic-jsk-rviz-plugins`, and common ROS/Gazebo dependencies.
- Python: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`, `torch`, `torchvision`, `casadi`.
- MPC solver: `acados` C library + `acados_template` Python interface, installed by the Dockerfile to `/opt/acados` (with `ACADOS_SOURCE_DIR`, `LD_LIBRARY_PATH`, and `PYTHONPATH` set).
- Initializes a catkin workspace at `/root/catkin_ws` and pre-builds it; `docker-compose.yml` mounts this repo into `/root/catkin_ws/src/assignment` for live edits.
