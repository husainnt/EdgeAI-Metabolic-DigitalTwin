"""
Quick comparison of the two candidate Patient 1031 dataset files, to figure
out whether results/cohort/patient_1031_merged.csv is a copy/superset of the
results/patient_1031_real_cgm_hr_steps_sleep*.csv chain, or something
unrelated.

Run this BEFORE running extract_and_merge_spo2.py's output through
training, and again AFTER you've placed the new SpO2 data wherever
window_generator.py actually reads from.

Usage:
    python check_cohort_file_consistency.py
"""

import os
import pandas as pd

# Update these two if your real filenames differ
CANDIDATE_A = "results/cohort/patient_1031_merged.csv"       # what window_generator.py defaults to
CANDIDATE_B = "results/patient_1031_real_cgm_hr_steps_sleep.csv"  # latest known chained file


def describe(path):
    if not os.path.exists(path):
        print(f"[!] {path} does NOT exist.")
        return None

    df = pd.read_csv(path)
    ts = pd.to_datetime(df["timestamp"]) if "timestamp" in df.columns else None

    print(f"\n--- {path} ---")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")
    if ts is not None:
        print(f"Date range: {ts.min()} to {ts.max()}")
    return df


def compare():
    df_a = describe(CANDIDATE_A)
    df_b = describe(CANDIDATE_B)

    if df_a is None or df_b is None:
        print("\n[!] Can't compare -- one of the two files is missing. "
              "Update CANDIDATE_A / CANDIDATE_B at the top of this script "
              "to point at whatever files actually exist on your machine.")
        return

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    cols_a, cols_b = set(df_a.columns), set(df_b.columns)
    print(f"Columns only in A ({CANDIDATE_A}): {cols_a - cols_b}")
    print(f"Columns only in B ({CANDIDATE_B}): {cols_b - cols_a}")
    print(f"Shared columns: {cols_a & cols_b}")

    print(f"\nRow count A: {len(df_a):,} | Row count B: {len(df_b):,}")
    if len(df_a) == len(df_b):
        print("[+] Same row count -- likely the same underlying grid, possibly just renamed/copied.")
    else:
        print("[!] DIFFERENT row counts -- these are NOT simply the same file under two names.")
        print("    Figure out which one is actually current/complete before proceeding.")

    if "heart_rate" in cols_a and "sleep_status" in cols_a:
        print("\n[+] Candidate A already has heart_rate + sleep_status -- some consolidation")
        print("    already happened at some point to produce this file.")
    else:
        print("\n[!] Candidate A is missing heart_rate and/or sleep_status -- this may NOT be")
        print("    the real, currently-used training file, or consolidation hasn't run recently.")


if __name__ == "__main__":
    compare()