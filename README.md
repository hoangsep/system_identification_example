# Polaris GEM System Identification & MPC

- Python ROS stack to record GEM simulation data, learn a neural dynamics model, and run an MPC using the learned model.
- Core scripts: `data_recorder.py` (logging), `train_model.py` (system ID), `validate_model.py`/`model_check.py` (validation), `mpc_controller.py` (controller).
- Trained artifacts included: `gem_dynamics.pth`, `gem_scaler.pkl`, `gem_scaler_arrays.npz`, and `sysid_validation_plot.png`.

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
- Launch Gazebo/RViz for the GEM simulator (example launch inside the bundled Polaris stack):  
  ```bash
  roslaunch polaris_gem_simulator gem_gazebo_rviz.launch
  ```
  (Adjust the launch file as needed for your scenario/track.)
- In another shell in the same container, run the controller from this repo:  
  ```bash
  source /root/catkin_ws/devel/setup.bash
  cd /root/catkin_ws/src/assignment
  python mpc_controller.py
  ```
  The node subscribes to `/gazebo/model_states` and `/gem/joint_states` and publishes `/gem/ackermann_cmd`.

## System Identification Workflow
- **Data capture**: Run the simulator, then log data with `python data_recorder.py` (from this repo with ROS environment sourced). Drive with `manual_driver.py` (keyboard) or `auto_driver.py` to excite the system. Logs go to `neo_data/*.csv` with `cmd_speed`, `cmd_steer`, `steer_actual`, `v_actual`, pose, and rates.
- **Training**: Fit the neural dynamics model and scalers:  
  ```bash
  python train_model.py
  ```
  Outputs: `gem_dynamics.pth`, `gem_scaler.pkl`, `gem_scaler_arrays.npz`, and `sysid_validation_plot.png` (predicted vs. actual Δx/Δy/Δyaw/Δv with per-signal RMSE in the titles; first 150 samples for readability). RMSE is printed in the console at the end of training.
- **Validation**: Quick quantitative check:  
  ```bash
  python model_check.py
  ```
  This reports RMSE and bias on a holdout split in physical units. For spot checks of dynamics responses, use `validate_model.py` (prints PASS/FAIL against a kinematic baseline).

## MPC Controller
- `mpc_controller.py` reconstructs the trained network inside CasADi and solves a horizon of size `HORIZON` with lookahead tuned by `PREVIEW_*`. Cost weights (`Q_lat`, `Q_vel`, etc.), rate limits (`MAX_ACCEL`, `MAX_STEER_RATE`), and safety coupling (`Q_safety`) are defined near the top of the file.
- The controller seeds with the latest measured steering (`steer_actual`), rolls out the learned model in the constraints, and publishes `AckermannDrive` commands at `DT` (0.1 s). Debug markers (`/gem/mpc_debug`) can be visualized in RViz when `PUBLISH_DEBUG_MARKERS=True`.

## Dockerfile (Dependencies)
- Base: `osrf/ros:noetic-desktop-full`.
- Apt: `ros-noetic-ackermann-msgs`, `ros-noetic-ros-control`, `ros-noetic-gazebo-ros-control`, `ros-noetic-jsk-rviz-plugins`, and common ROS/Gazebo dependencies.
- Python: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`, `torch`, `torchvision`, `casadi`.
- Initializes a catkin workspace at `/root/catkin_ws` and pre-builds it; `docker-compose.yml` mounts this repo into `/root/catkin_ws/src/assignment` for live edits.
