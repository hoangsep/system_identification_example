import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_oscillation(csv_path='mpc_debug.csv'):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    if len(df) < 2:
        print("Not enough data to plot.")
        return

    # Normalize time
    df['t'] = df['time'] - df['time'].iloc[0]

    plt.figure(figsize=(12, 18))

    # 1. Steering Command vs Average Actual
    plt.subplot(7, 1, 1)
    plt.plot(df['t'], df['steer_cmd'], label='Cmd', color='blue', linewidth=2)
    plt.plot(df['t'], df['steer_act'], label='Avg Act', color='red', linestyle='--', linewidth=2)
    plt.title("Steering: Command vs Average Actual")
    plt.ylabel("Steer (rad)")
    plt.legend()
    plt.grid(True)
    
    # 2. Wheel Analysis (Left vs Right)
    plt.subplot(7, 1, 2)
    if 'steer_left' in df.columns and 'steer_right' in df.columns:
        plt.plot(df['t'], df['steer_left'], label='Left', color='cyan', linestyle='-', alpha=0.8)
        plt.plot(df['t'], df['steer_right'], label='Right', color='magenta', linestyle='-', alpha=0.8)
        plt.plot(df['t'], df['steer_act'], label='Avg', color='black', linestyle=':', linewidth=1)
    plt.title("Wheel Synchronization: Left vs Right")
    plt.ylabel("Steer (rad)")
    plt.legend()
    plt.grid(True)

    # 3. CTE
    plt.subplot(7, 1, 3)
    plt.plot(df['t'], df['cte'], color='orange')
    plt.title("Cross Track Error (Signed)")
    plt.ylabel("Error (m)")
    plt.grid(True)
    
    # 4. Speed
    plt.subplot(7, 1, 4)
    plt.plot(df['t'], df['speed'], label='Actual', color='blue', linewidth=1.5)
    if 'cmd_speed' in df.columns:
        plt.plot(df['t'], df['cmd_speed'], label='Cmd', color='red', linestyle='--', alpha=0.7)
    plt.title("Speed: Command vs Actual")
    plt.ylabel("Speed (m/s)")
    plt.legend()
    plt.grid(True)
    
    # 5. Yaw Rate
    plt.subplot(7, 1, 5)
    plt.plot(df['t'], df['yaw_rate'], color='green', label='Odom Wz')
    if 'wz' in df.columns:
        plt.plot(df['t'], df['wz'], color='purple', linestyle='--', label='IMU Wz')
    plt.title("Yaw Rate (Odom vs IMU)")
    plt.ylabel("Rate (rad/s)")
    plt.legend()
    plt.grid(True)
    
    # 6. Acceleration (X/Y)
    plt.subplot(7, 1, 6)
    if 'ax' in df.columns:
        plt.plot(df['t'], df['ax'], label='Ax (Long)', color='blue', alpha=0.7)
        plt.plot(df['t'], df['ay'], label='Ay (Lat)', color='red', alpha=0.7)
    plt.title("IMU Acceleration (X/Y)")
    plt.ylabel("Acc (m/s^2)")
    plt.legend()
    plt.grid(True)
    
    # 7. Acceleration (Z) - Bump Detector
    plt.subplot(7, 1, 7)
    if 'az' in df.columns:
        plt.plot(df['t'], df['az'], label='Az (Vert)', color='black')
    plt.title("IMU Vertical Acceleration (Z)")
    plt.ylabel("Acc (m/s^2)")
    plt.xlabel("Time (s)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('oscillation_debug.png')
    print("Saved oscillation_debug.png")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else 'mpc_debug.csv'
    plot_oscillation(path)
