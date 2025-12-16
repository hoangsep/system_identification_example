import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def plot_path(csv_path='wps.csv'):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    try:
        df = pd.read_csv(csv_path, header=None)
    except Exception as e:
        print(f"Error reading csv: {e}")
        return

    # Check columns
    # Assumes Col 0: X, Col 1: Y, Col 2: Yaw
    if df.shape[1] < 3:
        print("CSV must have at least 3 columns (X, Y, Yaw)")
        return
        
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    yaw = df.iloc[:, 2].values

    plt.figure(figsize=(20, 20))
    
    # Plot the line
    plt.plot(x, y, 'b-', label='Path', linewidth=0.5, alpha=0.5)
    
    # Plot Quivers (Heading arrows)
    # Subsample to avoid clutter (e.g., every 10th point)
    step = 5 # Reduced step for more arrows, but made them clearer
    if len(x) < 100: step = 1
    
    # Calculate U, V components for arrows
    u = np.cos(yaw)
    v = np.sin(yaw)
    
    # High-Vis Arrows
    plt.quiver(x[::step], y[::step], u[::step], v[::step], 
               angles='xy', scale_units='xy', scale=0.2, color='r', width=0.0015, headwidth=4, label='Heading')
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'Path Visualization: {csv_path}')
    plt.axis('equal') # Important for correct aspect ratio
    plt.grid(True)
    plt.legend()
    
    output_file = 'path_debug.png'
    plt.savefig(output_file, dpi=300) # High DPI
    print(f"Saved {output_file} (High Res)")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else 'wps.csv'
    plot_path(path)
