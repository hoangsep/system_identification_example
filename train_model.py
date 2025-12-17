#!/usr/bin/env python3
import argparse
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR = Path("neo_data")

MODEL_SAVE_PATH = "gem_dynamics.pth"
SCALER_SAVE_PATH = "gem_scaler.pkl"
SCALER_ARRAY_PATH = "gem_scaler_arrays.npz"

SYSID_PLOT_PATH = "sysid_validation_plot.png"
SYSID_INPUT_OUTPUT_PLOT_PATH = "sysid_input_output.png"
RMSE_PLOT_PATH = "rmse_plot.png"

# Training Hyperparameters
EPOCHS = 8000
LEARNING_RATE = 0.005
PATIENCE = 150

# Controller Loop Settings
TARGET_DT = 0.05
DT_TOL = 0.01

# Weighting (recommended)
K_TURN = 20.0
K_ACCEL = 10.0
K_CMDERR = 10.0
WEIGHT_CAP = 50.0

# Subset thresholds (physical units)
TURN_DYAW_THRESH = 0.005      # rad per step (≈ 0.1 rad/s * 0.05s)
ACCEL_DV_THRESH = 0.02        # m/s per step (tune for your logs)
CMDERR_THRESH = 0.3           # |cmd_speed - v| in m/s (tune)


# ==========================================
# 2. DATA PREPROCESSING (Option A inputs)
# ==========================================
def wrap_angle_np(a):
    return np.arctan2(np.sin(a), np.cos(a))


def load_and_process_data(filename: Path):
    print(f"Loading data from {filename}...")
    df = pd.read_csv(filename)

    required_cols = [
        "time",
        "cmd_speed",
        "cmd_steer",
        "steer_actual",
        "x",
        "y",
        "yaw",
        "v_actual",
        "yaw_rate",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{filename} is missing: {missing}")

    # Determine stride to match TARGET_DT
    raw_dt = df["time"].diff().dropna()
    median_raw_dt = raw_dt.median()
    if median_raw_dt <= 0 or np.isnan(median_raw_dt):
        print(f"  -> Skipping {filename}: invalid time deltas.")
        return np.array([]), np.array([])

    stride = max(1, int(round(TARGET_DT / median_raw_dt)))
    print(f"  -> Raw dt: {median_raw_dt:.4f}s, Stride: {stride} (Target {TARGET_DT}s)")

    # Current, Next (t + stride), Prev (t - stride) aligned on raw indices
    df_curr = df.iloc[:-stride].reset_index(drop=True)
    df_next = df.shift(-stride).iloc[:-stride].reset_index(drop=True)
    df_prev = df.shift(stride).iloc[:-stride].reset_index(drop=True)

    dt = df_next["time"].values - df_curr["time"].values

    # Filter: DT validity + moving
    mask = (
        (dt > TARGET_DT - DT_TOL)
        & (dt < TARGET_DT + DT_TOL)
        & (df_curr["v_actual"].values > 0.01)
    )

    df_curr = df_curr[mask].reset_index(drop=True)
    df_next = df_next[mask].reset_index(drop=True)
    df_prev = df_prev[mask].reset_index(drop=True)
    dt = dt[mask]

    if len(df_curr) == 0:
        return np.array([]), np.array([])

    # -------- Inputs (Option A, 8D) --------
    # [v, steer_actual, yaw_rate, cmd_speed, cmd_steer, dt, prev_cmd_speed, prev_cmd_steer]
    X = np.column_stack(
        [
            df_curr["v_actual"].values,        # 0
            df_curr["steer_actual"].values,    # 1
            df_curr["yaw_rate"].values,        # 2
            df_curr["cmd_speed"].values,       # 3
            df_curr["cmd_steer"].values,       # 4
            dt,                                 # 5
            df_prev["cmd_speed"].values,       # 6 (prev)
            df_prev["cmd_steer"].values,       # 7 (prev)
        ]
    )

    # -------- Outputs (Targets, 5D) --------
    # [dx_local, dy_local, d_yaw, d_v, d_steer]
    dx_global = df_next["x"].values - df_curr["x"].values
    dy_global = df_next["y"].values - df_curr["y"].values
    yaw_curr = df_curr["yaw"].values

    d_yaw = df_next["yaw"].values - df_curr["yaw"].values
    d_yaw = wrap_angle_np(d_yaw)

    # Midpoint yaw rotation (recommended; matches MPC midpoint rotation)
    yaw_mid = yaw_curr + 0.5 * d_yaw
    dx_local = np.cos(yaw_mid) * dx_global + np.sin(yaw_mid) * dy_global
    dy_local = -np.sin(yaw_mid) * dx_global + np.cos(yaw_mid) * dy_global

    d_v = df_next["v_actual"].values - df_curr["v_actual"].values
    d_steer = df_next["steer_actual"].values - df_curr["steer_actual"].values

    Y = np.column_stack([dx_local, dy_local, d_yaw, d_v, d_steer])

    # Drop invalids (includes NaNs from df_prev for the first stride rows)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    X = X[valid]
    Y = Y[valid]

    return X, Y


def load_all_and_process(data_dir: Path):
    csv_files = sorted(Path(data_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    X_list, Y_list = [], []
    for csv_file in csv_files:
        X_part, Y_part = load_and_process_data(csv_file)
        if len(X_part) > 0:
            X_list.append(X_part)
            Y_list.append(Y_part)

    if not X_list:
        raise ValueError("No valid data found.")

    X_raw = np.vstack(X_list)
    Y_raw = np.vstack(Y_list)

    print(f"Total raw samples: {len(X_raw)}")
    return X_raw, Y_raw


def mirror_augment(X: np.ndarray, Y: np.ndarray):
    """
    Left/right mirroring in vehicle local/body frame.

    Inputs (8):
      [v, steer, yaw_rate, cmd_v, cmd_steer, dt, prev_cmd_v, prev_cmd_steer]
    Targets (5):
      [dx_local, dy_local, d_yaw, d_v, d_steer]
    """
    if X.size == 0 or Y.size == 0:
        return X, Y

    X_flip = X.copy()
    # flip steer-related signs
    X_flip[:, 1] *= -1.0   # steer_actual
    X_flip[:, 2] *= -1.0   # yaw_rate
    X_flip[:, 4] *= -1.0   # cmd_steer
    X_flip[:, 7] *= -1.0   # prev_cmd_steer

    Y_flip = Y.copy()
    Y_flip[:, 1] *= -1.0   # dy_local
    Y_flip[:, 2] *= -1.0   # d_yaw
    Y_flip[:, 4] *= -1.0   # d_steer

    return np.vstack([X, X_flip]), np.vstack([Y, Y_flip])


# ==========================================
# 3. MODEL
# ==========================================
class DynamicsModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 4. PLOTTING HELPERS
# ==========================================
def save_sysid_plots(actual: np.ndarray, pred: np.ndarray, d_yaw: np.ndarray, out_path: str):
    """
    Save a single image:
    - Top: sorted-by-d_yaw curves (actual vs pred)
    - Bottom: scatter actual vs pred with y=x line
    """
    if plt is None:
        print("[WARN] matplotlib not available; skipping sysid plot.")
        return

    labels = ["dX_local", "dY_local", "dYaw", "dV", "dSteer"]
    sort_idx = np.argsort(d_yaw)

    fig = plt.figure(figsize=(16, 10))

    # Top row: sorted curves
    for i in range(5):
        ax = plt.subplot(2, 5, i + 1)
        ax.plot(actual[sort_idx, i], label="Actual", alpha=0.7)
        ax.plot(pred[sort_idx, i], label="Pred", alpha=0.7)
        ax.set_title(f"{labels[i]} (sorted by dYaw)")
        ax.grid(True)
        if i == 0:
            ax.legend()

    # Bottom row: scatter
    for i in range(5):
        ax = plt.subplot(2, 5, 5 + i + 1)
        ax.scatter(actual[:, i], pred[:, i], s=5, alpha=0.3)
        lo = min(actual[:, i].min(), pred[:, i].min())
        hi = max(actual[:, i].max(), pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_title(f"{labels[i]} scatter")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved sysid plot to {out_path}")


def save_sysid_input_output_plot(
    X_raw: np.ndarray,
    actual: np.ndarray,
    pred: np.ndarray,
    out_path: str,
    *,
    max_points: int = 20000,
    seed: int = 0,
):
    """
    Save an "input vs output" scatter overview (Actual vs Pred).

    X_raw (8): [v, steer_actual, yaw_rate, cmd_speed, cmd_steer, dt, prev_cmd_speed, prev_cmd_steer]
    Y     (5): [dx_local, dy_local, d_yaw, d_v, d_steer]
    """
    if plt is None:
        print("[WARN] matplotlib not available; skipping sysid input-output plot.")
        return
    if X_raw.size == 0 or actual.size == 0 or pred.size == 0:
        print("[WARN] Empty arrays; skipping sysid input-output plot.")
        return
    if len(X_raw) != len(actual) or len(actual) != len(pred):
        raise ValueError(
            f"Input/output length mismatch: X={len(X_raw)} actual={len(actual)} pred={len(pred)}"
        )

    n = len(X_raw)
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        Xp = X_raw[idx]
        ap = actual[idx]
        pp = pred[idx]
    else:
        Xp, ap, pp = X_raw, actual, pred

    specs = [
        ("cmd_speed", Xp[:, 3], "dx_local", ap[:, 0], pp[:, 0]),
        ("cmd_steer", Xp[:, 4], "d_yaw", ap[:, 2], pp[:, 2]),
        ("cmd_speed", Xp[:, 3], "d_v", ap[:, 3], pp[:, 3]),
        ("yaw_rate", Xp[:, 2], "dy_local", ap[:, 1], pp[:, 1]),
        ("steer_actual", Xp[:, 1], "d_steer", ap[:, 4], pp[:, 4]),
        ("v_actual", Xp[:, 0], "d_yaw", ap[:, 2], pp[:, 2]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.ravel()
    for ax, (x_name, x, y_name, y_act, y_pred) in zip(axes, specs):
        ax.scatter(x, y_act, s=6, alpha=0.25, label="Actual")
        ax.scatter(x, y_pred, s=6, alpha=0.25, label="Pred")
        ax.set_title(f"{x_name} vs {y_name}")
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.grid(True)
        ax.legend(loc="best", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved sysid input-output plot to {out_path}")


def save_rmse_plot(rmse_all, rmse_turn, rmse_accel, out_path: str):
    if plt is None:
        print("[WARN] matplotlib not available; skipping RMSE plot.")
        return

    labels = ["dX", "dY", "dYaw", "dV", "dSteer"]
    x = np.arange(len(labels))

    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    width = 0.25
    ax.bar(x - width, rmse_all, width=width, label="All")
    ax.bar(x, rmse_turn, width=width, label="Turns")
    ax.bar(x + width, rmse_accel, width=width, label="Accel/Transient")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("RMSE (unscaled, physical units)")
    ax.grid(True, axis="y")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved RMSE plot to {out_path}")


# ==========================================
# 5. MAIN TRAINING (full-batch)
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument(
        "--scalers-only",
        action="store_true",
        help="Rebuild and save scalers then exit (no training).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load data
    X_raw, Y_raw = load_all_and_process(Path(args.data_dir))

    # 2) Split first (avoid leakage through augmentation/scaling)
    X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = train_test_split(
        X_raw, Y_raw, test_size=0.2, random_state=args.seed
    )

    # 3) Augment only training set
    X_train_raw, Y_train_raw = mirror_augment(X_train_raw, Y_train_raw)
    print(f"Train samples (aug): {len(X_train_raw)} | Test samples: {len(X_test_raw)}")

    # 4) Fit scalers on TRAIN only
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_x.fit_transform(X_train_raw)
    Y_train = scaler_y.fit_transform(Y_train_raw)
    X_test = scaler_x.transform(X_test_raw)
    Y_test = scaler_y.transform(Y_test_raw)

    with open(SCALER_SAVE_PATH, "wb") as f:
        pickle.dump({"x": scaler_x, "y": scaler_y}, f)
    np.savez(
        SCALER_ARRAY_PATH,
        x_mean=scaler_x.mean_,
        x_scale=scaler_x.scale_,
        y_mean=scaler_y.mean_,
        y_scale=scaler_y.scale_,
    )
    print(f"Scalers saved to {SCALER_SAVE_PATH} and {SCALER_ARRAY_PATH}")

    # dy_local sanity check
    dy_std = float(np.std(Y_train_raw[:, 1]))
    print(f"dy_local std (train raw): {dy_std:.6g}")

    if args.scalers_only:
        return

    # 5) Full-batch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32, device=device)

    # 6) Improved weights (turn + accel + cmd mismatch), computed from SCALED tensors
    # Turn intensity proxy: |d_yaw_scaled|   (target index 2)
    # Accel intensity proxy: |d_v_scaled|    (target index 3)
    # Cmd mismatch proxy: |cmd_speed_scaled - v_scaled| (input indices 3 and 0)
    with torch.no_grad():
        turn_intensity = torch.abs(Y_train_t[:, 2])
        accel_intensity = torch.abs(Y_train_t[:, 3])
        cmd_err = torch.abs(X_train_t[:, 3] - X_train_t[:, 0])

        w = 1.0 + (K_TURN * turn_intensity) + (K_ACCEL * accel_intensity) + (K_CMDERR * cmd_err)
        w = torch.clamp(w, 1.0, WEIGHT_CAP)
        w = w / w.mean()
        w = w.unsqueeze(1)

        print("Sample weights:")
        print(f"  min={w.min().item():.2f}, max={w.max().item():.2f}, mean={w.mean().item():.2f}")

    # 7) Model
    model = DynamicsModel(input_dim=8, output_dim=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion_raw = nn.MSELoss(reduction="none")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience, min_lr=1e-6
    )

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    EARLY_STOP_PAT = args.patience * 4  # generous, since full-batch is smooth

    # 8) Train (full-batch)
    print("\nStarting full-batch training (Option A inputs)...")
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        pred = model(X_train_t)
        loss_elements = criterion_raw(pred, Y_train_t)        # [N, 5]
        loss_weighted = (loss_elements * w).mean()

        loss_weighted.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_test_t)
            val_loss = nn.MSELoss()(val_pred, Y_test_t)

        scheduler.step(val_loss)

        v = float(val_loss.item())
        if v + 1e-12 < best_val:
            best_val = v
            best_state = {k: p.detach().cpu().clone() for k, p in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 100 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:5d} | train_w {loss_weighted.item():.6f} | val {v:.6f} | lr {lr_now:.6g}")

        if bad_epochs > EARLY_STOP_PAT:
            print(f"Early stopping (no improvement for {EARLY_STOP_PAT} epochs). Best val={best_val:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # ==========================================
    # 9) EVALUATION (unscaled)
    # ==========================================
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_test_t).detach().cpu().numpy()

    pred_real = scaler_y.inverse_transform(pred_scaled)
    actual_real = scaler_y.inverse_transform(Y_test)

    rmse_all = np.sqrt(np.mean((actual_real - pred_real) ** 2, axis=0))

    d_yaw_real = actual_real[:, 2]
    d_v_real = actual_real[:, 3]
    v_real = X_test_raw[:, 0]
    cmd_v_real = X_test_raw[:, 3]
    cmd_err_real = np.abs(cmd_v_real - v_real)

    turn_mask = np.abs(d_yaw_real) > TURN_DYAW_THRESH
    accel_mask = (np.abs(d_v_real) > ACCEL_DV_THRESH) | (cmd_err_real > CMDERR_THRESH)

    def safe_rmse(mask):
        if np.sum(mask) < 10:
            return np.full((5,), np.nan)
        return np.sqrt(np.mean((actual_real[mask] - pred_real[mask]) ** 2, axis=0))

    rmse_turn = safe_rmse(turn_mask)
    rmse_accel = safe_rmse(accel_mask)

    print("\n=== EVALUATION (RMSE, physical units) ===")
    labels = ["dX", "dY", "dYaw", "dV", "dSteer"]
    for i, lab in enumerate(labels):
        print(f"{lab:6s} | all={rmse_all[i]:.6g} | turn={rmse_turn[i]:.6g} | accel={rmse_accel[i]:.6g}")

    # ==========================================
    # 10) SAVE PLOTS
    # ==========================================
    save_sysid_plots(actual_real, pred_real, d_yaw_real, SYSID_PLOT_PATH)
    save_sysid_input_output_plot(X_test_raw, actual_real, pred_real, SYSID_INPUT_OUTPUT_PLOT_PATH)
    save_rmse_plot(rmse_all, rmse_turn, rmse_accel, RMSE_PLOT_PATH)


if __name__ == "__main__":
    main()
