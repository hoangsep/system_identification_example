import pandas as pd
import matplotlib.pyplot as plt

def analyze():
    try:
        df = pd.read_csv('mpc_debug.csv')
    except Exception as e:
        print(f"Error reading csv: {e}")
        return

    # Filter for moving
    df = df[df['speed'] > 0.1]
    
    # Split by steering command (Turn direction)
    left_turns = df[df['steer_cmd'] > 0.05]
    right_turns = df[df['steer_cmd'] < -0.05]
    straights = df[abs(df['steer_cmd']) <= 0.05]
    
    print(f"Overall Mean CTE: {df['cte'].mean():.4f}")
    print(f"Left Turns Mean CTE: {left_turns['cte'].mean():.4f} (Count: {len(left_turns)})")
    print(f"Right Turns Mean CTE: {right_turns['cte'].mean():.4f} (Count: {len(right_turns)})")
    print(f"Straights Mean CTE: {straights['cte'].mean():.4f} (Count: {len(straights)})")

if __name__ == "__main__":
    analyze()
