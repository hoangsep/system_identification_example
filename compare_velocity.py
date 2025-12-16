import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants from mpc.py
COL_X = 0
COL_Y = 1
COL_V = 4
MAX_REF_V = 3.0
MIN_REF_V = 0.5

def downsample_path(path, min_dist=0.1):
    if len(path) <= 1: return path
    downsampled = [path[0]]
    last_pt = path[0]
    for pt in path[1:]:
        dist = np.linalg.norm(pt[:2] - last_pt[:2])
        if dist >= min_dist:
            downsampled.append(pt)
            last_pt = pt
    return np.array(downsampled)

def calculate_velocity_profile(path):
    """
    Updates the velocity column (COL_V) based on path curvature.
    v_cmd = sqrt(a_lat / k)
    """
    # 1. Compute Gradients
    x = path[:, COL_X]
    y = path[:, COL_Y]
    
    # Central difference for 1st n 2nd derivatives
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # 2. Compute Curvature k = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    num = np.abs(dx * ddy - dy * ddx)
    den = np.power(dx**2 + dy**2, 1.5) + 1e-6
    k = num / den
    
    # 3. Compute Velocity Limit
    max_lat_accel = 1.5 # m/s^2 (Tuned for stability)
    
    print(f"Max Curvature: {np.max(k):.4f}")
    print(f"Min Radius: {1.0/(np.max(k)+1e-6):.4f} m")
    
    new_v = np.zeros(len(path))
    
    for i in range(len(path)):
        curvature = k[i]
        # v^2/r = a_lat => v = sqrt(a_lat * r) = sqrt(a_lat / k)
        v_limit = np.sqrt(max_lat_accel / (curvature + 1e-3))
        
        # Apply Limit (Smoothly)
        v_final = min(MAX_REF_V, v_limit)
        
        # Enforce Min Speed
        v_final = max(v_final, MIN_REF_V)
    # ... (loop)
        new_v[i] = v_final
        
    return new_v, k

def main():
    # Load and process
    df = pd.read_csv('wps.csv', header=None)
    raw_path = df.values
    
    # Original data
    original_v = raw_path[:, COL_V]
    
    # Process
    full_path_processed = downsample_path(raw_path, min_dist=0.1)
    
    # Calculate
    calculated_v, k = calculate_velocity_profile(full_path_processed)
    
    # Start plotting
    plt.figure(figsize=(12, 10))
    
    # Subplot 1: Velocity
    plt.subplot(2, 1, 1)
    # Plot 1: Original Velocity
    plt.plot(np.linspace(0, 1, len(original_v)), original_v, label='Original wps.csv Velocity', color='red', alpha=0.5)

    
    # Plot 2: Calculated Velocity
    plt.plot(np.linspace(0, 1, len(calculated_v)), calculated_v, label='Calculated Dynamic Profile', color='blue', linewidth=2)
    
    plt.title('Velocity Profile Comparison')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Path Geometry
    plt.subplot(2, 1, 2)
    plt.plot(full_path_processed[:, COL_X], full_path_processed[:, COL_Y], label='Path', color='black')
    plt.title(f'Path Geometry (Min Radius: {1.0/(np.max(k)+1e-6):.1f}m)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.grid(True)
    
    output_file = 'velocity_comparison.png'
    plt.savefig(output_file)
    print(f"Comparison saved to {output_file}")
    
    # Print stats
    print(f"\nStats:")
    print(f"Original Max V: {np.max(original_v):.4f} m/s")
    print(f"Original Mean V: {np.mean(original_v):.4f} m/s")
    print(f"Calculated Max V: {np.max(calculated_v):.4f} m/s")
    print(f"Calculated Mean V: {np.mean(calculated_v):.4f} m/s")

if __name__ == "__main__":
    main()
