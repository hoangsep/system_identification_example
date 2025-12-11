#!/usr/bin/env python3
"""
Compute the maximum absolute acceleration and steering rate across all log CSVs in neo_data.

Acceleration is derived from successive differences of v_actual over time; steering rate is taken
from the recorded steer_rate column when present, otherwise derived from steer_actual.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def compute_metrics(csv_path: Path):
    df = pd.read_csv(csv_path)
    required = ['time', 'v_actual', 'steer_actual']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    time = df['time'].to_numpy()
    v_actual = df['v_actual'].to_numpy()

    dt = np.diff(time)
    dv = np.diff(v_actual)
    acc_mask = np.isfinite(dt) & np.isfinite(dv) & (dt > 0)
    acc = dv[acc_mask] / dt[acc_mask]
    max_abs_acc = float(np.max(np.abs(acc))) if acc.size else float('nan')

    if 'steer_rate' in df.columns:
        steer_rate_raw = df['steer_rate'].to_numpy()
        sr_mask = np.isfinite(steer_rate_raw)
        steer_rate = steer_rate_raw[sr_mask]
    else:
        # Fallback: differentiate steer_actual if steer_rate was not logged
        steer = df['steer_actual'].to_numpy()
        dsteer = np.diff(steer)
        sr_mask = np.isfinite(dsteer) & (dt > 0)
        steer_rate = dsteer[sr_mask] / dt[sr_mask]

    max_abs_sr = float(np.max(np.abs(steer_rate))) if steer_rate.size else float('nan')
    return max_abs_acc, max_abs_sr


def main():
    parser = argparse.ArgumentParser(description="Compute max accel and steering rate from neo_data CSV logs.")
    parser.add_argument("--data-dir", type=Path, default=Path("neo_data"), help="Directory containing CSV logs.")
    args = parser.parse_args()

    csv_files = sorted(args.data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {args.data_dir}")

    max_acc_values = []
    max_sr_values = []

    for csv_file in csv_files:
        max_acc, max_sr = compute_metrics(csv_file)
        print(f"{csv_file.name}: max|accel|={max_acc:.4f} m/s^2, max|steer_rate|={max_sr:.4f} rad/s")
        if np.isfinite(max_acc):
            max_acc_values.append(max_acc)
        if np.isfinite(max_sr):
            max_sr_values.append(max_sr)

    if max_acc_values:
        print(f"Overall max|accel| across logs: {max(max_acc_values):.4f} m/s^2")
    if max_sr_values:
        print(f"Overall max|steer_rate| across logs: {max(max_sr_values):.4f} rad/s")


if __name__ == "__main__":
    main()
