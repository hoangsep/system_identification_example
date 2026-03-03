import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_trajectory(pickle_path='mpc_trajectories.pkl', target_time=85.0):
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"File {pickle_path} not found.")
        return

    # Find the nearest frame to target_time
    times = [d['time'] for d in data]
    start_time = times[0]
    normalized_times = np.array(times) - start_time
    
    idx = (np.abs(normalized_times - target_time)).argmin()
    frame = data[idx]
    actual_time = normalized_times[idx]
    
    print(f"Plotting Frame at T={actual_time:.2f}s (Index {idx})")
    
    ref = frame['ref'] # Shape (4, Horizon) -> x, y, yaw, v
    pred = frame['pred'] # Shape (7, Horizon+1) -> x, y, yaw, v, steer, rate, cmd
    
    plt.figure(figsize=(10, 10))
    
    # Plot Trajectories
    plt.plot(ref[0, :], ref[1, :], 'g.--', label='Reference', markersize=10)
    plt.plot(pred[0, :], pred[1, :], 'b.-', label='Predicted', markersize=10)
    
    # Plot Start Point
    plt.plot(pred[0, 0], pred[1, 0], 'ro', label='Car', markersize=12)
    
    # Quiver for heading
    plt.quiver(pred[0, :], pred[1, :], np.cos(pred[2, :]), np.sin(pred[2, :]), color='blue', alpha=0.5)
    
    plt.title(f"MPC Trajectory at T={actual_time:.2f}s")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('trajectory_debug.png')
    print("Saved trajectory_debug.png")

if __name__ == "__main__":
    t = float(sys.argv[1]) if len(sys.argv) > 1 else 85.0
    plot_trajectory(target_time=t)
