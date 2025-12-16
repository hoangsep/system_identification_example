import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Configuration (Match MPC)
COL_X = 0
COL_Y = 1
COL_V = 4
MAX_REF_V = 3.0
MIN_REF_V = 0.5
MAX_LAT_ACCEL = 0.4 # Reduced to trigger on 20m turns

def calculate_profile(path):
    # 1. Least Squares Circle Fit (5-Point Window for Noise Reduction)
    k = np.zeros(len(path))
    window = 5 # Use 5 points centered on i
    half = window // 2
    
    for i in range(half, len(path) - half):
        # Extract local points
        pts = path[i-half : i+half+1, :2] # Shape (5, 2)
        
        # Center points to reduce numerical errors
        centroid = np.mean(pts, axis=0)
        pts_c = pts - centroid
        x = pts_c[:, 0]
        y = pts_c[:, 1]
        
        # Formulate Least Squares: x^2 + y^2 = A*x + B*y + C
        # M * [A, B, C]^T = D
        # where A=2xc, B=2yc, C=R^2 - xc^2 - yc^2
        
        M = np.column_stack((x, y, np.ones(len(x))))
        D = x**2 + y**2
        
        try:
            # Solve linear system
            params, _, _, _ = np.linalg.lstsq(M, D, rcond=None)
            A, B, C = params
            
            xc = A / 2.0
            yc = B / 2.0
            R_sq = C + xc**2 + yc**2
            R = np.sqrt(max(R_sq, 1e-6))
            
            # Curvature k = 1/R
            k[i] = 1.0 / (R + 1e-6)
        except np.linalg.LinAlgError:
            k[i] = 0.0
            
    # Pad endpoints
    k[:half] = k[half]
    k[-half:] = k[-half-1]
    
    # 3. Compute Profile
    v_profile = np.zeros_like(k)
    
    print(f"DEBUG: Max Curvature: {np.max(k):.4f}, Mean: {np.mean(k):.4f}")
    
    for i in range(len(path)):
        # v = sqrt(a_lat / k)
        raw_limit = np.sqrt(MAX_LAT_ACCEL / (k[i] + 1e-4))
        v_limit = min(MAX_REF_V, raw_limit)
        v_limit = max(v_limit, MIN_REF_V)
        v_profile[i] = v_limit
        
    print(f"DEBUG: V_Profile Min: {np.min(v_profile):.4f}, Max: {np.max(v_profile):.4f}")
    print(f"DEBUG: Raw Limit Sample (first 5): {[f'{x:.2f}' for x in np.sqrt(MAX_LAT_ACCEL / (k[:5] + 1e-4))]}")
        
    return k, v_profile

def downsample_path(path, min_dist=0.1):
    if len(path) <= 1: return path
    downsampled = [path[0]]
    last_p = path[0, :2]
    for i in range(1, len(path)):
        curr_p = path[i, :2]
        dist = np.linalg.norm(curr_p - last_p)
        if dist >= min_dist:
            downsampled.append(path[i])
            last_p = curr_p
    return np.array(downsampled)

def plot_velocity_profile(csv_path='wps.csv'):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    try:
        df = pd.read_csv(csv_path, header=None)
        raw_path = df.values
    except Exception as e:
        print(f"Error reading csv: {e}")
        return

    # Downsample first (Critical for gradient calculation)
    # Testing robust resolution (0.2)
    path = downsample_path(raw_path, min_dist=0.2)
    
    # Calculate
    k, v_profile = calculate_profile(path)
    
    # Distance vector s
    diffs = np.diff(path[:, [COL_X, COL_Y]], axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg_lens)))
    
    # Plotting
    plt.figure(figsize=(12, 10))
    
    # 1. Path Colored by Suggested Speed
    plt.subplot(2, 2, 1)
    sc = plt.scatter(path[:, COL_X], path[:, COL_Y], c=v_profile, cmap='jet', s=3)
    plt.colorbar(sc, label='Target Speed (m/s)')
    plt.title("Path colored by Calculated Speed Profile")
    plt.axis('equal')
    plt.grid(True)
    
    # 2. Curvature vs Velocity (Dual Axis)
    plt.subplot(2, 1, 2)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Left Axis: Curvature
    lns1 = ax1.plot(s, k, color='purple', alpha=0.6, label='Curvature (1/m)')
    ax1.set_xlabel("Distance along path (m)")
    ax1.set_ylabel("Curvature (1/m)", color='purple')
    ax1.tick_params(axis='y', labelcolor='purple')
    ax1.grid(True)

    # Right Axis: Velocity
    lns2 = ax2.plot(s, v_profile, label='Calc Limit', color='blue', linestyle='--', alpha=0.5)
    
    lns3 = []
    if path.shape[1] > COL_V:
        lns3 = ax2.plot(s, path[:, COL_V], label='File Velocity', color='orange', linewidth=2)
        
    ax2.set_ylabel("Speed (m/s)", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    # Auto-Scale Y-Axis
    max_v = MAX_REF_V
    if path.shape[1] > COL_V:
        max_v = max(max_v, np.max(path[:, COL_V]))
    ax2.set_ylim(0, max_v + 1.0)
    
    # Legend
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper center')
    
    plt.title("Correlation: Curvature vs Velocity")
    
    plt.tight_layout()
    plt.savefig('velocity_profile_debug.png')
    print("Saved velocity_profile_debug.png")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else 'wps.csv'
    plot_velocity_profile(path)
