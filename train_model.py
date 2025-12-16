import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR = Path('neo_data')
MODEL_SAVE_PATH = 'gem_dynamics.pth'
SCALER_SAVE_PATH = 'gem_scaler.pkl'
SCALER_ARRAY_PATH = 'gem_scaler_arrays.npz'
PLOT_SAVE_PATH = 'sysid_validation_plot.png'

# Training Hyperparameters
EPOCHS = 8000
BATCH_SIZE = 128
LEARNING_RATE = 0.005
PATIENCE = 150 # Increased patience for weighted optimization

# Weighting Factor: How much more important is a turn than a straight line?
TURN_WEIGHT_FACTOR = 20.0 

# Controller Loop Settings
TARGET_DT = 0.05
DT_TOL = 0.01

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================
def load_and_process_data(filename):
    print(f"Loading data from {filename}...")
    df = pd.read_csv(filename)

    required_cols = ['time', 'cmd_speed', 'cmd_steer', 'steer_actual', 
                     'x', 'y', 'yaw', 'v_actual', 'yaw_rate']
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{filename} is missing: {missing}")

    # Resample / Stride Calculation
    raw_dt = df['time'].diff().dropna()
    median_raw_dt = raw_dt.median()
    if median_raw_dt <= 0 or np.isnan(median_raw_dt):
        print(f"Skipping {filename}: Invalid time deltas.")
        return np.array([]), np.array([])

    stride = max(1, int(round(TARGET_DT / median_raw_dt)))
    print(f"  -> Raw dt: {median_raw_dt:.4f}s, Stride: {stride} (Target {TARGET_DT}s)")
    
    # Create 'Next' DataFrame (State t+1)
    df_curr = df.iloc[:-stride].reset_index(drop=True)
    df_next = df.shift(-stride).iloc[:-stride].reset_index(drop=True)
    
    # Filter by DT validity and minimal movement
    dt = df_next['time'].values - df_curr['time'].values
    mask = (
        (dt > TARGET_DT - DT_TOL) & 
        (dt < TARGET_DT + DT_TOL) & 
        (df_curr['v_actual'].values > 0.01)
    )

    df_curr = df_curr[mask].reset_index(drop=True)
    df_next = df_next[mask].reset_index(drop=True)
    dt = dt[mask]

    if len(df_curr) == 0:
        return np.array([]), np.array([])

    # --- Inputs (Features) ---
    # [v, steer_actual, yaw_rate, cmd_speed, cmd_steer, dt]
    X = np.column_stack([
        df_curr['v_actual'].values,      # 0
        df_curr['steer_actual'].values,  # 1
        df_curr['yaw_rate'].values,      # 2
        df_curr['cmd_speed'].values,     # 3
        df_curr['cmd_steer'].values,     # 4
        dt                               # 5
    ])

    # --- Outputs (Targets) ---
    # [dx_local, dy_local, d_yaw, d_v, d_steer]
    
    # Global Deltas
    dx_global = df_next['x'].values - df_curr['x'].values
    dy_global = df_next['y'].values - df_curr['y'].values
    yaw_curr = df_curr['yaw'].values

    # Rotate to Local Frame
    dx_local = np.cos(yaw_curr) * dx_global + np.sin(yaw_curr) * dy_global
    dy_local = -np.sin(yaw_curr) * dx_global + np.cos(yaw_curr) * dy_global
    
    # Delta Yaw (wrapped)
    d_yaw = df_next['yaw'].values - df_curr['yaw'].values
    d_yaw = np.arctan2(np.sin(d_yaw), np.cos(d_yaw))
    
    d_v = df_next['v_actual'].values - df_curr['v_actual'].values
    d_steer = df_next['steer_actual'].values - df_curr['steer_actual'].values

    Y = np.column_stack([dx_local, dy_local, d_yaw, d_v, d_steer])
    
    valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    return X[valid_mask], Y[valid_mask]

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
    Left/right mirroring in the vehicle (local/body) frame.

    Inputs:  [v, steer_actual, yaw_rate, cmd_speed, cmd_steer, dt]
    Targets: [dx_local, dy_local, d_yaw, d_v, d_steer]
    """
    if X.size == 0 or Y.size == 0:
        return X, Y

    X_flip = X.copy()
    X_flip[:, 1] *= -1.0  # steer_actual
    X_flip[:, 2] *= -1.0  # yaw_rate
    X_flip[:, 4] *= -1.0  # cmd_steer

    Y_flip = Y.copy()
    Y_flip[:, 1] *= -1.0  # dy_local
    Y_flip[:, 2] *= -1.0  # d_yaw
    Y_flip[:, 4] *= -1.0  # d_steer

    X_aug = np.vstack([X, X_flip])
    Y_aug = np.vstack([Y, Y_flip])
    return X_aug, Y_aug

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
class DynamicsModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DynamicsModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. TRAINING ROUTINE
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    X_raw, Y_raw = load_all_and_process(DATA_DIR)

    # 2. Split FIRST (avoid test leakage via augmentation/scaling)
    X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = train_test_split(
        X_raw, Y_raw, test_size=0.2, random_state=42
    )

    # 3. Augment ONLY the training set (mirroring)
    X_train_raw, Y_train_raw = mirror_augment(X_train_raw, Y_train_raw)
    print(f"Train samples (after augmentation): {len(X_train_raw)} | Test samples: {len(X_test_raw)}")

    # 4. Scale using TRAIN statistics only
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train = scaler_x.fit_transform(X_train_raw)
    Y_train = scaler_y.fit_transform(Y_train_raw)
    X_test = scaler_x.transform(X_test_raw)
    Y_test = scaler_y.transform(Y_test_raw)

    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump({'x': scaler_x, 'y': scaler_y}, f)
    
    np.savez(SCALER_ARRAY_PATH,
             x_mean=scaler_x.mean_, x_scale=scaler_x.scale_,
             y_mean=scaler_y.mean_, y_scale=scaler_y.scale_)
    print("Scalers saved.")

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32, device=device)

    # =========================================================
    # 4. WEIGHT CALCULATION (Fixing the Imbalance)
    # =========================================================
    # We use the scaled d_yaw (Index 2 in Y) as a proxy for turn intensity.
    # Since it's scaled, values > 1.0 are significant turns.
    
    print("Calculating Sample Weights...")
    # Calculate absolute magnitude of the turn in the training set
    turn_intensity = torch.abs(Y_train_t[:, 2]) 
    
    # Weight = 1.0 (Base) + Factor * Intensity
    # If d_yaw is 0 (Straight), Weight = 1.0
    # If d_yaw is 2 sigma (Sharp Turn), Weight = 1.0 + 20 * 2 = 41.0
    train_weights = 1.0 + (TURN_WEIGHT_FACTOR * turn_intensity)
    
    # Normalize weights so the mean is 1.0 (keeps learning rate consistent)
    train_weights = train_weights / train_weights.mean()
    train_weights = train_weights.unsqueeze(1) # [Batch, 1] for broadcasting
    
    print(f"  -> Min Weight: {train_weights.min().item():.2f}")
    print(f"  -> Max Weight: {train_weights.max().item():.2f}")

    # 5. Init Model
    model = DynamicsModel(input_dim=6, output_dim=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Important: reduction='none' so we can apply weights element-wise
    criterion_raw = nn.MSELoss(reduction='none') 
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=PATIENCE, min_lr=1e-6
    )

    # 6. Train
    print("\nStarting Training with Weighted Loss...")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        output = model(X_train_t)
        
        # Calculate element-wise loss [Batch, 5]
        loss_elements = criterion_raw(output, Y_train_t)
        
        # Apply weights across all output dimensions for each sample
        # (We weight the whole sample based on how hard the turn is)
        loss_weighted = (loss_elements * train_weights).mean()
        
        loss_weighted.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_out = model(X_test_t)
            # Use standard MSE for validation to track true error
            val_loss = nn.MSELoss()(val_out, Y_test_t)
            
        scheduler.step(val_loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Train (Weighted) {loss_weighted.item():.5f}, Val (Raw) {val_loss.item():.5f}, LR {optimizer.param_groups[0]['lr']:.6f}")
            if val_loss.item() < 0.0025 and optimizer.param_groups[0]['lr'] < 1e-4:
                print("Converged.")
                break

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # =========================================================
    # 7. EVALUATION (The True Test)
    # =========================================================
    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_test_t).detach().cpu().numpy()
    
    pred_real = scaler_y.inverse_transform(pred_scaled)
    actual_real = scaler_y.inverse_transform(Y_test)

    # Metric 1: Global RMSE
    rmse_global = np.sqrt(np.mean((actual_real - pred_real)**2, axis=0))
    
    # Metric 2: TURN RMSE (The important one!)
    # Define a turn as yaw_rate > 0.1 rad/s (approx 6 deg/s)
    # Note: We look at the actual physical YAW RATE (from input X, which we need to recover or infer)
    # Easier way: Look at Y_test (d_yaw).
    
    # Recover unscaled d_yaw for the test set
    d_yaw_test_real = actual_real[:, 2]
    
    # Define threshold for "Turning"
    # 0.1 rad/s * 0.05s = 0.005 rad change per step
    turn_mask = np.abs(d_yaw_test_real) > 0.005 
    
    print("\n=== EVALUATION RESULTS ===")
    print(f"Global RMSE (dYaw):  {rmse_global[2]:.5f} rad")
    
    if np.sum(turn_mask) > 0:
        rmse_turn = np.sqrt(np.mean((actual_real[turn_mask] - pred_real[turn_mask])**2, axis=0))
        print(f"TURN RMSE (dYaw):    {rmse_turn[2]:.5f} rad  <-- THIS MUST BE LOW")
        print(f"TURN RMSE (dLat):    {rmse_turn[1]:.5f} m")
    else:
        print("No turns found in test set to evaluate.")

    # Plot
    plt.figure(figsize=(15, 10))
    
    # Sort by turning magnitude for cleaner plotting
    sort_idx = np.argsort(d_yaw_test_real)
    
    labels = ['dX', 'dY', 'dYaw', 'dV', 'dSteer']
    for i in range(5):
        plt.subplot(2, 3, i+1)
        # Plot only a subset of sorted data to see the "S" curve of prediction
        plt.plot(actual_real[sort_idx, i], label='Actual', color='black', alpha=0.6)
        plt.plot(pred_real[sort_idx, i], label='Pred', color='red', alpha=0.6)
        plt.title(labels[i])
        plt.legend()
        plt.grid(True)
        
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH)
    print(f"Plot saved to {PLOT_SAVE_PATH}")
