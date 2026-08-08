import pandas as pd
import matplotlib.pyplot as plt

from gem_mpc import paths

def analyze(csv_path=paths.RESULTS_DIR / "mpc_debug.csv"):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading csv: {e}")
        return

    # Filter for moving
    df = df[df['speed'] > 0.1]
    
    # Split by steering command (Turn direction)
    left_turns = df[df['steer_cmd_pub'] > 0.05]
    right_turns = df[df['steer_cmd_pub'] < -0.05]
    straights = df[abs(df['steer_cmd_pub']) <= 0.05]
    
    print(f"Overall Mean CTE: {df['cte_signed'].mean():.4f}")
    print(f"Left Turns Mean CTE: {left_turns['cte_signed'].mean():.4f} (Count: {len(left_turns)})")
    print(f"Right Turns Mean CTE: {right_turns['cte_signed'].mean():.4f} (Count: {len(right_turns)})")
    print(f"Straights Mean CTE: {straights['cte_signed'].mean():.4f} (Count: {len(straights)})")

if __name__ == "__main__":
    analyze()
