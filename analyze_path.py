import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_path(csv_path='wps.csv'):
    try:
        df = pd.read_csv(csv_path, header=None)
        path = df.values
        print(f"Loaded path with {len(path)} points.")
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return

    # Check for NaNs
    if np.isnan(path).any():
        print("WARNING: Path contains NaNs!")

    # Calculate distances between consecutive points
    diffs = np.diff(path[:, 0:2], axis=0) # X, Y
    dists = np.linalg.norm(diffs, axis=1)
    
    # Statistics
    print(f"Min segment length: {np.min(dists):.4f}")
    print(f"Max segment length: {np.max(dists):.4f}")
    print(f"Mean segment length: {np.mean(dists):.4f}")
    print(f"Std segment length: {np.std(dists):.4f}")

    # Find jumps
    JUMP_THRESHOLD = 0.5 # meters
    jump_indices = np.where(dists > JUMP_THRESHOLD)[0]
    if len(jump_indices) > 0:
        print(f"\nWARNING: Found {len(jump_indices)} jumps > {JUMP_THRESHOLD}m:")
        for idx in jump_indices:
            print(f"  Index {idx} -> {idx+1}: Dist = {dists[idx]:.4f}m at ({path[idx,0]:.2f}, {path[idx,1]:.2f})")

    # Find duplicates (dist approx 0)
    DUPLICATE_THRESHOLD = 0.001
    dup_indices = np.where(dists < DUPLICATE_THRESHOLD)[0]
    if len(dup_indices) > 0:
        print(f"\nWARNING: Found {len(dup_indices)} duplicate/extremely close points (< {DUPLICATE_THRESHOLD}m).")
        print("  (This can cause numerical issues with orientation calculation)")

    # Plot segment lengths
    plt.figure(figsize=(10, 6))
    plt.plot(dists)
    plt.title("Path Segment Lengths")
    plt.xlabel("Index")
    plt.ylabel("Distance (m)")
    plt.savefig("path_analysis.png")
    print("\nSaved path_analysis.png")

if __name__ == "__main__":
    analyze_path()
