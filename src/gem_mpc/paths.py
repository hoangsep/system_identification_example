"""Repository-anchored filesystem paths.

Every path is derived from this file's location rather than the working
directory, so scripts behave the same no matter where they are launched from.
Set GEM_MPC_ROOT to override the root (needed only for a non-editable install).
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("GEM_MPC_ROOT", Path(__file__).resolve().parents[2]))

# recorded driving logs consumed by training
DATA_DIR = ROOT / "data"

# trained network weights and input/output scalers
MODELS_DIR = ROOT / "models"

# run outputs: debug logs, trajectory dumps, generated plots
RESULTS_DIR = ROOT / "results"

# reference path followed by the controller
WAYPOINTS_CSV = ROOT / "waypoints" / "wps.csv"

# acados C code generation target, kept out of the repo root
ACADOS_BUILD = ROOT / "build" / "acados"
ACADOS_OCP_JSON = ACADOS_BUILD / "acados_ocp.json"

MODEL_PATH = MODELS_DIR / "gem_dynamics.pth"
SCALER_PATH = MODELS_DIR / "gem_scaler.pkl"
SCALER_ARRAY_PATH = MODELS_DIR / "gem_scaler_arrays.npz"


def result(name: str) -> Path:
    """Return a path inside results/, creating the directory on first use."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name


def acados_build() -> Path:
    """Return the acados code export directory, creating it if needed."""
    ACADOS_BUILD.mkdir(parents=True, exist_ok=True)
    return ACADOS_BUILD
