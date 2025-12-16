import pandas as pd
import sys

def analyze_log(csv_path='mpc_debug.csv'):
    try:
        df = pd.read_csv(csv_path)
    except:
        print("No file.")
        return

    # Find first row where Abs(CTE) > 1.0
    divergence = df[df['cte'].abs() > 1.0]
    
    if len(divergence) == 0:
        print("CTE never exceeded 1.0m.")
        return
    
    first_idx = divergence.index[0]
    print(f"Divergence found at index {first_idx} (Time {df.iloc[first_idx]['time']})")
    
    # Show 20 rows before
    start = max(0, first_idx - 20)
    end = min(len(df), first_idx + 5)
    
    print(df.iloc[start:end])

if __name__ == "__main__":
    analyze_log()
