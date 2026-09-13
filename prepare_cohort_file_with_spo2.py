"""
Consolidates the SpO2-merged Patient 1031 file into the actual training
path (results/cohort/patient_1031_merged.csv), while DELIBERATELY dropping
the 'steps' column for now.

WHY drop 'steps' here: window_generator.py's has_act check is
"'steps' in df.columns" -- if we copy this file over with 'steps' still in
it, Activity silently flips from off to on in the SAME run that adds SpO2,
confounding two changes into one result. The 'steps' column's data quality
has NOT been validated yet (see inspect_activity_data_quality.py / the
Debug_activity_schema.py empty-string finding) -- so we test SpO2 in
isolation first, then validate and add Activity as its own deliberate step.

This script also runs a quick sanity check on the steps column's *value
distribution* while it's here (even though it drops it), purely so you have
that data point on hand for later -- it does NOT gate anything in this run.

Usage:
    python prepare_cohort_file_with_spo2.py
"""

import os
import shutil
import pandas as pd

SPO2_MERGED_INPUT = "results/patient_1031_real_cgm_hr_steps_sleep_spo2.csv"
COHORT_TARGET = "results/cohort/patient_1031_merged.csv"
BACKUP_TARGET = "results/cohort/patient_1031_merged_backup_pre_spo2.csv"


def quick_steps_sanity_check(df):
    if "steps" not in df.columns:
        return
    s = df["steps"]
    n = len(s)
    n_zero = (s == 0).sum()
    n_null = s.isna().sum()
    n_nonzero = n - n_zero - n_null
    print("\n--- steps column sanity check (informational only, not gating this run) ---")
    print(f"Total rows: {n:,} | Zero: {n_zero:,} ({100*n_zero/n:.1f}%) | "
          f"Non-zero: {n_nonzero:,} ({100*n_nonzero/n:.1f}%) | Null: {n_null:,}")
    if n_nonzero > 0:
        print(f"Non-zero steps -- mean: {s[s > 0].mean():.1f}, max: {s.max():.1f}")
    print("(This column is being DROPPED from the cohort file for this run --")
    print(" see script docstring for why.)")


def consolidate():
    if not os.path.exists(SPO2_MERGED_INPUT):
        print(f"[!] {SPO2_MERGED_INPUT} not found. Run extract_and_merge_spo2.py first,")
        print("    or update SPO2_MERGED_INPUT at the top of this script.")
        return

    if os.path.exists(COHORT_TARGET):
        shutil.copy2(COHORT_TARGET, BACKUP_TARGET)
        print(f"[+] Backed up existing cohort file to: {BACKUP_TARGET}")
    else:
        print(f"[i] No existing file at {COHORT_TARGET} to back up.")

    df = pd.read_csv(SPO2_MERGED_INPUT)
    print(f"[i] Loaded {SPO2_MERGED_INPUT}: {len(df):,} rows, columns: {df.columns.tolist()}")

    if "oxygen_saturation" not in df.columns:
        raise RuntimeError(
            "[!] CRITICAL: 'oxygen_saturation' column not found in the SpO2-merged "
            "input. Did extract_and_merge_spo2.py actually run successfully?"
        )

    quick_steps_sanity_check(df)

    if "steps" in df.columns:
        df = df.drop(columns=["steps"])
        print("[i] Dropped 'steps' column deliberately (Activity stays off this round).")

    os.makedirs(os.path.dirname(COHORT_TARGET), exist_ok=True)
    df.to_csv(COHORT_TARGET, index=False)
    print(f"[SAVED] {COHORT_TARGET} now has columns: {df.columns.tolist()}")
    print("\n[+] Ready to train with real HR + Sleep + SpO2 + Calibration-Context.")
    print("    Activity remains honestly off until validated and added as its own step.")


if __name__ == "__main__":
    consolidate()