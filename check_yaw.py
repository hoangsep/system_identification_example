import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def check_yaw(csv_path='wps.csv'):
    df = pd.read_csv(csv_path, header=None)
    path = df.values
    
    # Downsample logic from mpc.py
    min_dist = 0.1
    downsampled = [path[0]]
    last_p = path[0, :2]
    for i in range(1, len(path)):
        curr_p = path[i, :2]
        dist = np.linalg.norm(curr_p - last_p)
        if dist >= min_dist:
            downsampled.append(path[i])
            last_p = curr_p
    path = np.array(downsampled)
    
    yaw = path[:, 2] # COL_YAW
    
    # Check for jumps
    diffs = np.diff(yaw)
    # Wrap diffs
    diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
    
    plt.figure(figsize=(10,6))
    plt.plot(yaw, label='Yaw')
    plt.title("Path Yaw (Downsampled)")
    plt.savefig("yaw_check.png")
    
    max_jump = np.max(np.abs(diffs))
    print(f"Max Yaw Jump: {max_jump:.4f} rad")
    
    # Find index of max jump
    idx = np.argmax(np.abs(diffs))
    print(f"Max Jump at Index {idx}: {diffs[idx]:.4f} rad")
    print(f"Yaw ends: {yaw[idx]} -> {yaw[idx+1]}")

if __name__ == "__main__":
    check_yaw()
