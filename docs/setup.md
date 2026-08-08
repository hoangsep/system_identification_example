# Setup

Two ways to run this project. The Docker image is the only path that gives you the
full stack (ROS Noetic, Gazebo, acados); the host install covers training and
offline analysis.

## Docker (full stack)

The image is built from [Dockerfile](../Dockerfile) on top of
`osrf/ros:noetic-desktop-full` and adds:

- ROS packages: `ackermann-msgs`, `ros-control`, `ros-controllers`,
  `gazebo-ros-control`, `hector-gazebo-plugins`, `joy`, `jsk-rviz-plugins`
- Python packages from [requirements.txt](../requirements.txt)
- [acados](https://github.com/acados/acados) built from source into `/opt/acados`,
  with `ACADOS_SOURCE_DIR`, `LD_LIBRARY_PATH` and `PYTHONPATH` set
- A catkin workspace at `/root/catkin_ws`, pre-built

### Build and start

```bash
docker compose build

xhost +local:root          # let the container reach your X server
docker compose up -d
docker exec -it gem_mpc_container bash
```

[docker-compose.yml](../docker-compose.yml) mounts this repo at
`/root/catkin_ws/src/assignment`, so edits on the host are live in the container.
It also requests an NVIDIA GPU; if you do not have `nvidia-container-toolkit`
installed, comment out the `deploy:` block.

### Build the workspace

Needed once per container, and again whenever the simulator packages change:

```bash
cd /root/catkin_ws
catkin_make
source devel/setup.bash
```

### Run

```bash
# terminal 1: simulator
roslaunch gem_gazebo gem_gazebo_rviz.launch

# terminal 2: controller
python3 -m gem_mpc.mpc
```

No install step is needed inside the container: the Dockerfile puts
`/root/catkin_ws/src/assignment/src` on `PYTHONPATH`.

## Host install (training and analysis only)

`train_model.py`, `validate_model.py` and everything in `src/gem_mpc/tools/` need
only numpy, pandas, matplotlib, scipy, scikit-learn and torch. They do not import
ROS or acados, so they run on a plain Python environment:

```bash
pip install -e .
gem-train
gem-validate
python -m gem_mpc.tools.inspect_data
```

`mpc.py` and `data_recorder.py` will not import without ROS.

## How paths are resolved

[src/gem_mpc/paths.py](../src/gem_mpc/paths.py) derives every path from its own file
location, so scripts behave identically no matter which directory you launch them
from:

| Constant | Location |
|---|---|
| `paths.DATA_DIR` | `data/` |
| `paths.MODELS_DIR` | `models/` |
| `paths.RESULTS_DIR` | `results/` |
| `paths.WAYPOINTS_CSV` | `waypoints/wps.csv` |
| `paths.ACADOS_BUILD` | `build/acados/` |

The one case this cannot handle is a non-editable install, where the package is
copied into `site-packages` and no longer sits inside the repo. Set `GEM_MPC_ROOT`
to the repo root if you hit that:

```bash
export GEM_MPC_ROOT=/path/to/system_identification_example
```

## Generated files

Nothing below is tracked; all of it is reproducible.

| Path | Produced by |
|---|---|
| `build/acados/` | acados C code generation and `acados_ocp.json`, on first `mpc.py` run |
| `results/*.png` | Plots from `train_model.py`, `mpc.py` and the tools |
| `results/mpc_debug.csv` | Per-step controller log, written on `mpc.py` shutdown |
| `results/mpc_trajectories.pkl` | Reference and predicted horizons per step, same shutdown hook |

The trajectory pickle grows quickly (hundreds of MB for a long session) because it
stores the full horizon at every control step. Delete it between runs if you are
not using `tools/plot_trajectory.py`.
