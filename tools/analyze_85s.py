import pandas as pd
import numpy as np

def analyze_85s(csv_path='mpc_debug.csv'):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Normalize time
    df['t'] = df['time'] - df['time'].iloc[0]
    
    # Filter window
    window = df[(df['t'] >= 84.0) & (df['t'] <= 87.0)]
    
    if len(window) == 0:
        print("No data found around 85s. Total duration: {:.2f}s".format(df['t'].iloc[-1]))
        return

    print(f"\nAnalysis Window: 84.0s to 87.0s ({len(window)} samples)")
    print("-" * 100)
    print(f"{'Time':<8} {'CTE':<8} {'Cmd':<8} {'AvgAct':<8} {'L_Act':<8} {'R_Act':<8} {'YawRate':<8} {'Az':<8} {'Wz':<8}")
    print("-" * 100)
    
    for i, row in window.iterrows():
        print(f"{row['t']:<8.3f} {row['cte']:<8.4f} {row['steer_cmd']:<8.4f} {row['steer_act']:<8.4f} {row['steer_left']:<8.4f} {row['steer_right']:<8.4f} {row['yaw_rate']:<8.4f} {row['az']:<8.4f} {row['wz']:<8.4f}")

    # Check for spikes
    print("\nStatistics in Window:")
    print(window[['cte', 'steer_cmd', 'steer_act', 'az', 'wz']].describe())

if __name__ == "__main__":
    analyze_85s()
