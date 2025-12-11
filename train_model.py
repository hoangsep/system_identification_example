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
DATA_DIR = Path('data')  # train on all CSVs in this folder
MODEL_SAVE_PATH = 'gem_dynamics.pth'
SCALER_SAVE_PATH = 'gem_scaler.pkl'
SCALER_ARRAY_PATH = 'gem_scaler_arrays.npz'
PLOT_SAVE_PATH = 'sysid_validation_plot.png'

# Training Hyperparameters
EPOCHS = 8000          # Increased epochs since we have a scheduler
BATCH_SIZE = 128       # Larger batch size for stability
LEARNING_RATE = 0.005  # Start with a higher learning rate
PATIENCE = 100         # How many epochs to wait before lowering LR
# We train the dynamics to match the controller loop (20 Hz).
# The raw log was recorded at ~1 kHz, so we resample to ~0.05 s steps.
TARGET_DT = 0.1
DT_TOL = 0.01  # keep samples whose dt is within +/- this window

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================
def load_and_process_data(filename):
    print(f"Loading data from {filename}...")
    df = pd.read_csv(filename)

    # Raw logs were collected at ~1 kHz. Pick a stride so that the "next"
    # sample is roughly TARGET_DT away from the current one.
    raw_dt = df['time'].diff().dropna()
    median_raw_dt = raw_dt.median()
    stride = max(1, int(round(TARGET_DT / median_raw_dt)))
    print(f"Median raw dt: {median_raw_dt:.4f}s, stride: {stride} -> target ~{TARGET_DT:.3f}s")
    
    # Create "Next State" columns by shifting the dataframe up by the stride
    df_next = df.shift(-stride)
    
    # Calculate dt (Time step) and drop the last `stride` rows (NaN due to shift)
    df = df.iloc[:-stride]
    df_next = df_next.iloc[:-stride]
    dt = df_next['time'].values - df['time'].values
    
    # Keep samples that match the controller loop timing and avoid bad data
    mask = (
        (dt > TARGET_DT - DT_TOL) &
        (dt < TARGET_DT + DT_TOL) &
        (df['v_actual'].values > 0.05)
    )
    
    # Apply filtering
    df = df[mask].reset_index(drop=True)
    df_next = df_next[mask].reset_index(drop=True)
    dt = dt[mask]

    if len(df) == 0:
        raise ValueError(f"No usable samples in {filename} after resampling/filtering")

    print(f"Data count after filtering: {len(df)} samples (dt mean {dt.mean():.4f}s, std {dt.std():.4f}s)")


    # --- 1. INPUTS (Features) ---
    # [v_current, cmd_speed, cmd_steer, dt]
    # Adding 'dt' is CRITICAL so the model learns Physics (Dist = Vel * Time)
    X = np.column_stack([
        df['v_actual'].values,
        df['cmd_speed'].values,
        df['cmd_steer'].values,
        dt
    ])

    # --- 2. OUTPUTS (Targets) ---
    # We predict the CHANGE in state (Delta), not the absolute position
    
    # Calculate Global Deltas
    dx_global = df_next['x'].values - df['x'].values
    dy_global = df_next['y'].values - df['y'].values
    yaw = df['yaw'].values

    # Rotate Global Deltas into Local Frame
    # Because the physics of the car are the same whether it's at x=0 or x=100
    dx_local = np.cos(yaw) * dx_global + np.sin(yaw) * dy_global
    dy_local = -np.sin(yaw) * dx_global + np.cos(yaw) * dy_global
    
    # Calculate Yaw Delta (handle -pi to pi wrap-around)
    d_yaw = df_next['yaw'].values - df['yaw'].values
    d_yaw = np.arctan2(np.sin(d_yaw), np.cos(d_yaw))
    
    # Calculate Velocity Delta
    d_v = df_next['v_actual'].values - df['v_actual'].values

    # Target Vector: [dx_local, dy_local, d_yaw, d_v]
    Y = np.column_stack([dx_local, dy_local, d_yaw, d_v])
    
    return X, Y


def load_all_and_process(data_dir: Path):
    csv_files = sorted(Path(data_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    X_list, Y_list = [], []
    for csv_file in csv_files:
        X_part, Y_part = load_and_process_data(csv_file)
        if len(X_part) == 0:
            print(f"Skipping {csv_file} (no samples after filtering)")
            continue
        X_list.append(X_part)
        Y_list.append(Y_part)

    if not X_list:
        raise ValueError("No samples remained after filtering any CSV files.")

    X_raw = np.vstack(X_list)
    Y_raw = np.vstack(Y_list)

    # --- AUGMENTATION ---
    # We want the model to know that Left Turn = -Right Turn.
    # We flip all signs associated with "Y" axis and "Yaw".
    # X columns: [v, cmd_v, cmd_steer, dt]
    # Y columns: [dx_local, dy_local, d_yaw, d_v]
    
    # Create flipped copy
    X_flip = X_raw.copy()
    X_flip[:, 2] *= -1.0  # Flip cmd_steer

    Y_flip = Y_raw.copy()
    Y_flip[:, 1] *= -1.0  # Flip dy_local
    Y_flip[:, 2] *= -1.0  # Flip d_yaw
    
    # Concatenate original + flipped
    X_final = np.vstack([X_raw, X_flip])
    Y_final = np.vstack([Y_raw, Y_flip])
    
    print(f"Total samples after augmentation: {len(X_final)} (from {len(X_raw)} original)")

    return X_final, Y_final

# ==========================================
# 3. IMPROVED NEURAL NETWORK
# ==========================================
class DynamicsModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DynamicsModel, self).__init__()
        
        # Smaller network for faster CasADi graph / MPC solve
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
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Data ---
    X_raw, Y_raw = load_all_and_process(DATA_DIR)

    # --- Normalize Data ---
    # Neural Networks fail if inputs are 20.0 and 0.001 mixed together.
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_x.fit_transform(X_raw)
    Y_scaled = scaler_y.fit_transform(Y_raw)

    # Save scalers for the MPC controller later
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump({'x': scaler_x, 'y': scaler_y}, f)
        print(f"Scalers saved to {SCALER_SAVE_PATH}")
    np.savez(SCALER_ARRAY_PATH,
             x_mean=scaler_x.mean_, x_scale=scaler_x.scale_,
             y_mean=scaler_y.mean_, y_scale=scaler_y.scale_)
    print(f"Scaler arrays saved to {SCALER_ARRAY_PATH}")

    # Split Train/Test
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

    # Convert to PyTorch Tensors on the chosen device
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32, device=device)

    # --- Initialize Model ---
    # Input Dim = 4 (v, cmd_v, cmd_s, dt)
    # Output Dim = 4 (dx, dy, dtheta, dv)
    model = DynamicsModel(input_dim=4, output_dim=4).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Scheduler: Reduce LR if validation loss doesn't improve
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=PATIENCE, min_lr=1e-5
    )

    # --- Training Loop ---
    print("\nStarting Training...")
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, Y_train_t)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_test_t)
            val_loss = criterion(val_out, Y_test_t)
            
        # Update Scheduler
        scheduler.step(val_loss)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        if epoch % 100 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}: Train Loss {loss.item():.5f}, Val Loss {val_loss.item():.5f}, LR {lr:.6f}")
            
            # Early stopping check (optional manually)
            if val_loss.item() < 0.005: 
                print("Converged! Stopping early.")
                break

    # --- Save Model ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

    # --- Evaluation & Plotting ---
    model.eval()
    with torch.no_grad():
        predicted_scaled = model(X_test_t).detach().cpu().numpy()
    
    # Inverse transform to get real physical units
    predicted_real = scaler_y.inverse_transform(predicted_scaled)
    actual_real = scaler_y.inverse_transform(Y_test)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((actual_real - predicted_real)**2, axis=0))
    print("\n=== Model Accuracy (RMSE) ===")
    print(f"dX (local): {rmse[0]:.4f} m")
    print(f"dY (local): {rmse[1]:.4f} m")
    print(f"dYaw:      {rmse[2]:.4f} rad")
    print(f"dV:        {rmse[3]:.4f} m/s")

    # Plot Results
    plt.figure(figsize=(14, 10))
    labels = ['Delta X (Local)', 'Delta Y (Local)', 'Delta Yaw', 'Delta Velocity']
    
    # Only plot first 150 points to make the graph readable
    limit = 150 
    
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(actual_real[:limit, i], label='Actual', color='black', alpha=0.6)
        plt.plot(predicted_real[:limit, i], label='Predicted', color='red', alpha=0.7, linestyle='--')
        plt.title(f"{labels[i]} - RMSE: {rmse[i]:.4f}")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH)
    print(f"Validation plot saved as {PLOT_SAVE_PATH}")
