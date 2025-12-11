import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

DATA_DIR = Path('data')

def analyze_steering():
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    all_steer = []
    
    print(f"Found {len(csv_files)} logs.")
    
    for f in csv_files:
        df = pd.read_csv(f)
        steer = df['cmd_steer'].values
        all_steer.extend(steer)
        print(f"  - {f.name}: {len(df)} samples, Mean Steer: {np.mean(steer):.3f}")

    all_steer = np.array(all_steer)
    
    # Stats
    total = len(all_steer)
    zero_mask = (np.abs(all_steer) < 0.05)
    zeros = np.sum(zero_mask)
    
    print("\n=== DATA DISTRIBUTION ===")
    print(f"Total Samples: {total}")
    print(f"Straight (Steer < 0.05): {zeros} ({zeros/total*100:.1f}%)")
    print(f"Turning  (Steer >= 0.05): {total-zeros} ({(total-zeros)/total*100:.1f}%)")
    
    plt.figure()
    plt.hist(all_steer, bins=50, log=True)
    plt.title("Steering Command Distribution (Log Scale)")
    plt.xlabel("Steer Angle (rad)")
    plt.ylabel("Count")
    plt.savefig("data_distribution.png")
    print("Saved histogram to data_distribution.png")

if __name__ == "__main__":
    analyze_steering()
