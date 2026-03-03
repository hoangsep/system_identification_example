import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import sys

def test_spline_fit(csv_path='wps.csv'):
    # Load Data
    try:
        df = pd.read_csv(csv_path, header=None)
        path = df.values # [x, y, yaw, v, ...]
    except:
        print("wps.csv not found")
        return

    x = path[:, 0]
    y = path[:, 1]
    
    # 1. Fit Spline
    # s is smoothing factor. 0 = interpolate through all points.
    # We want some smoothing to remove noise.
    # splprep returns: tck (knots, coeffs, degree), u (parameter values)
    
    # Remove duplicates which cause errors
    diff = np.diff(path[:, :2], axis=0)
    dist = np.linalg.norm(diff, axis=1)
    valid_mask = np.concatenate(([True], dist > 0.01))
    x = x[valid_mask]
    y = y[valid_mask]
    
    # Fit
    # s=0.5 -> Amount of smoothness. Tune this.
    try:
        tck, u = splprep([x, y], s=0.5, per=0) 
    except Exception as e:
        print(f"Spline fit failed: {e}")
        return

    # 2. Evaluate at fine resolution
    u_fine = np.linspace(0, 1, num=2000)
    x_new, y_new = splev(u_fine, tck)
    
    # 3. Calculate Curvature Analytically
    # First derivatives
    dx, dy = splev(u_fine, tck, der=1)
    # Second derivatives
    ddx, ddy = splev(u_fine, tck, der=2)
    
    k = (dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Path
    plt.subplot(2, 1, 1)
    plt.plot(x, y, 'r.', markersize=2, label='Raw Points')
    plt.plot(x_new, y_new, 'b-', linewidth=1.5, label='Spline Fit')
    plt.title("Spline Fit vs Raw Data")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    
    # Curvature
    plt.subplot(2, 1, 2)
    plt.plot(u_fine, k, 'g-', label='Spline Curvature')
    plt.title("Analytical Curvature from Spline")
    plt.ylim(-0.2, 0.2)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('spline_debug.png')
    print("Saved spline_debug.png")

if __name__ == "__main__":
    test_spline_fit()
