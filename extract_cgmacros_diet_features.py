"""
Extracts real diet-event features for the 14 CGMacros T2D patients (subject
IDs confirmed via derive_cgmacros_t2d_subset.py against the published
paper's own A1c-based classification).

Design mirrors the existing glc_context pattern (time-since-calibration +
last calibration value) rather than inventing something new -- diet is an
EVENT signal (43/14,730 rows populated for a sample participant), not a
continuous sequence, so it belongs in an MLP-based EventModalityEncoder
matching enc_calib_glucose's pattern, not an LSTM.

For every row in the grid, computes:
    time_since_last_meal_min -- minutes since the most recent logged meal
    carbs_eaten, protein_eaten, fat_eaten, fiber_eaten, calories_eaten --
        the most recent meal's macros, each scaled by that meal's own
        "Amount Consumed" percentage (a meal logged as only 60% eaten
        contributes 60% of its logged macros, not the full logged amount)
    has_had_meal -- 0 before the first meal in the record, 1 after

This gives enc_diet 7 genuine real features (5 scaled macros +
time_since_last_meal_min + has_had_meal) vs the architecture's current
in_features=3 placeholder -- widening enc_diet to match is a separate,
one-line change in hybrid_twin.py, not done in this script.

Usage:
    python extract_cgmacros_diet_features.py
"""

import os
import pandas as pd
import numpy as np

CGMACROS_ROOT = r"D:\FYP\DATA_SET\cgmacros-a-scientific-dataset-for-personalized-nutrition-and-diet-monitoring-1.0.0\CGMacros_dateshifted365\CGMacros"
T2D_SUBJECT_LIST_CSV = "results/cgmacros_t2d_subject_list.csv"
OUTPUT_DIR = "results/cgmacros_cohort"

MACRO_COLUMNS = ["Carbs", "Protein", "Fat", "Fiber", "Calories"]


def extract_diet_features(pid: int):
    csv_path = os.path.join(CGMACROS_ROOT, f"CGMacros-{pid:03d}", f"CGMacros-{pid:03d}.csv")
    if not os.path.exists(csv_path):
        print(f"[!] {csv_path} not found, skipping subject {pid}")
        return None

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    has_meal_row = df["Meal Type"].notna()
    n_meals = has_meal_row.sum()
    print(f"[i] Subject {pid}: {len(df)} rows, {n_meals} logged meals")

    # Here I don't assume 'Amount Consumed' exists with that exact name in
    # every participant's file -- subject 28 broke this assumption, so I
    # check first and report the real columns if it's missing, rather than
    # silently defaulting or crashing the whole batch
    if "Amount Consumed" in df.columns:
        amount_frac = (df["Amount Consumed"].fillna(100) / 100.0)
    else:
        print(f"[!] Subject {pid}: 'Amount Consumed' column not found. Real columns: {df.columns.tolist()}")
        print(f"    Defaulting to 100% consumed for this patient -- FLAG THIS in your report as a")
        print(f"    disclosed per-patient schema gap, not a silently-assumed value.")
        amount_frac = pd.Series(1.0, index=df.index)
    for col in MACRO_COLUMNS:
        df[f"{col}_eaten"] = np.where(has_meal_row, df[col] * amount_frac, np.nan)

    # Here I forward-fill each meal's scaled macros to every row until the
    # next meal, giving every timestep "what was the most recent meal"
    eaten_cols = [f"{c}_eaten" for c in MACRO_COLUMNS]
    df[eaten_cols] = df[eaten_cols].ffill().fillna(0.0)

    # Here I compute minutes since the most recent meal, and whether any
    # meal has happened yet at all (has_had_meal=0 for the very start of
    # the record, before the first logged meal)
    meal_times = df.loc[has_meal_row, "Timestamp"]
    time_since_last_meal_min = np.full(len(df), np.nan)
    has_had_meal = np.zeros(len(df), dtype=np.float32)

    last_meal_time = None
    for i, row in df.iterrows():
        if has_meal_row.iloc[i]:
            last_meal_time = row["Timestamp"]
        if last_meal_time is not None:
            time_since_last_meal_min[i] = (row["Timestamp"] - last_meal_time).total_seconds() / 60.0
            has_had_meal[i] = 1.0

    df["time_since_last_meal_min"] = time_since_last_meal_min
    # Here I fill the pre-first-meal period with a large placeholder value
    # (24 hours) rather than leaving NaN -- consistent with "no real recent
    # meal signal yet" rather than an undefined number reaching the model
    df["time_since_last_meal_min"] = df["time_since_last_meal_min"].fillna(24 * 60.0)
    df["has_had_meal"] = has_had_meal

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"cgmacros_{pid:03d}_diet_features.csv")

    desired_cols = ["Timestamp", "Libre GL", "Dexcom GL", "HR", "Calories (Activity)", "METs",
                     "time_since_last_meal_min", "has_had_meal"] + eaten_cols
    keep_cols = [c for c in desired_cols if c in df.columns]
    missing_cols = [c for c in desired_cols if c not in df.columns]
    if missing_cols:
        print(f"[!] Subject {pid}: missing columns {missing_cols} -- saving without them. "
              f"FLAG as a disclosed per-patient schema gap in your report.")

    df[keep_cols].to_csv(out_path, index=False)
    print(f"[SAVED] {out_path}")
    return out_path


if __name__ == "__main__":
    subj_df = pd.read_csv(T2D_SUBJECT_LIST_CSV)
    t2d_subjects = subj_df[subj_df["is_t2d"]]["subject"].tolist()
    print(f"[i] Processing {len(t2d_subjects)} T2D subjects: {t2d_subjects}\n")

    succeeded, failed = [], []
    for pid in t2d_subjects:
        try:
            result = extract_diet_features(pid)
            if result:
                succeeded.append(pid)
            else:
                failed.append(pid)
        except Exception as e:
            print(f"[!] Subject {pid} FAILED with: {e}")
            print(f"    Skipping this patient, continuing with the rest of the batch.")
            failed.append(pid)
            continue

    print(f"\n{'='*60}")
    print(f"BATCH SUMMARY: {len(succeeded)}/{len(t2d_subjects)} succeeded")
    print(f"  Succeeded: {succeeded}")
    if failed:
        print(f"  Failed entirely: {failed}")
    print(f"{'='*60}")