"""
Deep-dive diagnostic for Patients 7751 and 7188 -- both have unremarkable
glucose stats (mean ~100-110 mg/dL, std ~25-30) yet scored physics RMSE in
the 160-175 mg/dL range, similar to patients averaging 300+ mg/dL. This
mismatch suggests a physics-engine warm-start problem specific to these two
patients, not genuine clinical severity.

Plots real glucose vs the simglucose warm-started baseline side by side, so
you can see exactly WHERE the simulation diverges (right after a specific
calibration point? drifting over the whole series? one bad outlier?).

Usage:
    python inspect_standout_patients.py
"""

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "glycemic_twin", "ml_layer")))
from window_generator import extract_2x_daily_calibrations, compute_simglucose_baseline

STANDOUT_PATIENTS = ["7751", "7188", "7392"]
COHORT_CSV_PATTERN = "results/cohort/patient_{pid}_merged.csv"


def inspect(pid: str):
    csv_path = COHORT_CSV_PATTERN.format(pid=pid)
    if not os.path.exists(csv_path):
        print(f"[!] {csv_path} not found, skipping {pid}")
        return

    df = pd.read_csv(csv_path)
    calib_indices = extract_2x_daily_calibrations(df)
    sim_baseline, tsc_series, calib_series = compute_simglucose_baseline(df, calib_indices)

    real = df["glucose_mg_dl"].values.astype(float)
    err = np.abs(real - sim_baseline)
    worst_idx = np.argsort(err)[-10:][::-1]

    print(f"\n=== Patient {pid} ===")
    print(f"Calibration points: {len(calib_indices)}")
    print(f"Overall RMSE (physics vs real): {np.sqrt(np.mean((real - sim_baseline)**2)):.1f}")
    print("Worst 10 individual timesteps (index, real, sim, abs_error):")
    for i in worst_idx:
        print(f"  idx={i:5d}  real={real[i]:6.1f}  sim={sim_baseline[i]:6.1f}  |err|={err[i]:6.1f}")

    plt.figure(figsize=(14, 5))
    plt.plot(real, label="Real glucose", color="black", linewidth=1)
    plt.plot(sim_baseline, label="simglucose warm-started baseline", color="red", alpha=0.7)
    for ci in calib_indices:
        plt.axvline(ci, color="blue", alpha=0.15, linewidth=0.8)
    plt.title(f"Patient {pid}: Real vs Physics-Only Baseline (blue lines = calibration points)")
    plt.xlabel("5-min timestep")
    plt.ylabel("Glucose (mg/dL)")
    plt.legend()
    plt.tight_layout()
    out_path = f"results/diagnostic_{pid}_real_vs_sim.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    for pid in STANDOUT_PATIENTS:
        inspect(pid)