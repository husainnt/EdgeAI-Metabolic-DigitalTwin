"""
Consolidates the Activity-merged Patient 1031 file into the actual training
path (results/cohort/patient_1031_merged.csv), preserving the SpO2 column
already merged in by prepare_cohort_file_with_spo2.py.

Usage:
    python prepare_cohort_file_with_activity.py
"""

import os
import shutil
import pandas as pd

ACTIVITY_MERGED_INPUT = "results/patient_1031_real_cgm_hr_steps_sleep_spo2_activity.csv"
COHORT_TARGET = "results/cohort/patient_1031_merged.csv"
BACKUP_TARGET = "results/cohort/patient_1031_merged_backup_pre_activity.csv"

REQUIRED_NEW_COLUMNS = ["steps_sum", "activity_walking_frac"]


def consolidate():
    if not os.path.exists(ACTIVITY_MERGED_INPUT):
        print(f"[!] {ACTIVITY_MERGED_INPUT} not found. Run extract_and_merge_activity.py first,")
        print("    or update ACTIVITY_MERGED_INPUT at the top of this script.")
        return

    if os.path.exists(COHORT_TARGET):
        shutil.copy2(COHORT_TARGET, BACKUP_TARGET)
        print(f"[+] Backed up existing cohort file to: {BACKUP_TARGET}")
    else:
        print(f"[i] No existing file at {COHORT_TARGET} to back up.")

    df = pd.read_csv(ACTIVITY_MERGED_INPUT)
    print(f"[i] Loaded {ACTIVITY_MERGED_INPUT}: {len(df):,} rows, columns: {df.columns.tolist()}")

    missing = [c for c in REQUIRED_NEW_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"[!] CRITICAL: expected column(s) {missing} not found in the Activity-merged "
            "input. Did extract_and_merge_activity.py actually run successfully?"
        )

    if "oxygen_saturation" not in df.columns:
        print("[!] WARNING: 'oxygen_saturation' column not present in this file -- SpO2 may")
        print("    have been lost somewhere in the chain. Double-check INPUT_CSV in")
        print("    extract_and_merge_activity.py pointed at the SpO2-included cohort file.")

    os.makedirs(os.path.dirname(COHORT_TARGET), exist_ok=True)
    df.to_csv(COHORT_TARGET, index=False)
    print(f"[SAVED] {COHORT_TARGET} now has columns: {df.columns.tolist()}")
    print("\n[+] Ready to train with real HR + Sleep + SpO2 + Activity + Calibration-Context.")


if __name__ == "__main__":
    consolidate()