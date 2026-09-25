"""
Extracts real diet-event features for the CGMacros T2D patients.

CHANGED this pass (2026-09): two real data-quality problems found by
cross-checking every extracted meal against PhysioNet's own
DataDictionary_CGMacros-00X.csv documented per-meal ranges:
    1. Several subjects have raw Carbs/Protein/Fat/Fiber/Calories values
       far outside the documented range (e.g. subject 39: 522g carbs in
       one meal, documented max is 176g). These are capped at the
       documented max BEFORE Amount-Consumed scaling, and every capped
       meal is flagged in a new 'macro_capped' column -- the glucose
       curve around that meal is kept, only the implausible macro INPUT
       is capped, disclosed, not silently trusted or silently dropped.
    2. Subject 30's entire meal record is 10-100x smaller than every
       other subject's (max carbs 7.2g vs a ~40-95g population range) --
       a unit-scale mismatch, not a per-meal outlier. Excluded entirely
       via EXCLUDE_SUBJECTS, same pattern as this project's
       EXCLUDE_PATIENT_IDS convention elsewhere.

Design still mirrors the existing glc_context pattern (time-since-
calibration + last calibration value) -- diet is an EVENT signal, not a
continuous sequence, so it belongs in an MLP-based EventModalityEncoder
matching enc_calib_glucose's pattern, not an LSTM.

For every row in the grid, computes:
    time_since_last_meal_min -- minutes since the most recent logged meal
    carbs_eaten, protein_eaten, fat_eaten, fiber_eaten, calories_eaten --
        the most recent meal's macros (capped, then scaled by that meal's
        own Amount Consumed percentage)
    has_had_meal -- 0 before the first meal in the record, 1 after
    macro_capped -- 1 if this meal's macros hit the documented-range cap
        at least once, 0 otherwise (forward-filled with the meal, so a
        capped meal's whole "recent meal" window carries the flag)

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

# Here I cap at PhysioNet's own DataDictionary_CGMacros-00X.csv documented
# per-meal ranges -- values outside these are logging/parsing artifacts,
# not real physiological quantities, per the dataset's own definition
MACRO_CAPS = {"Carbs": 176.0, "Protein": 176.0, "Fat": 176.0, "Fiber": 176.0, "Calories": 1180.0}

# Here I exclude subject 30 -- its whole meal record is 10-100x smaller
# than every other subject's, a unit-scale mismatch rather than a
# per-meal outlier, so capping individual values would just guess at a
# wrong number instead of fixing the real problem
EXCLUDE_SUBJECTS = {30}


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

    # Here I cap each raw macro column at its documented max BEFORE the
    # Amount-Consumed scaling, and record which meal rows got capped
    capped_any = pd.Series(False, index=df.index)
    for col in MACRO_COLUMNS:
        cap = MACRO_CAPS[col]
        over_cap = has_meal_row & (df[col] > cap)
        n_over = int(over_cap.sum())
        if n_over:
            print(f"[!] Subject {pid}: {n_over} meal(s) had {col} above the documented max "
                  f"({cap:g}) -- capping to {cap:g} and flagging. Example raw values: "
                  f"{df.loc[over_cap, col].round(1).tolist()[:5]}")
            df.loc[over_cap, col] = cap
            capped_any |= over_cap

    if "Amount Consumed" in df.columns:
        amount_frac = (df["Amount Consumed"].fillna(100) / 100.0)
    else:
        print(f"[!] Subject {pid}: 'Amount Consumed' column not found. Real columns: {df.columns.tolist()}")
        print(f"    Defaulting to 100% consumed for this patient -- FLAG THIS in your report as a")
        print(f"    disclosed per-patient schema gap, not a silently-assumed value.")
        amount_frac = pd.Series(1.0, index=df.index)
    for col in MACRO_COLUMNS:
        df[f"{col}_eaten"] = np.where(has_meal_row, df[col] * amount_frac, np.nan)

    eaten_cols = [f"{c}_eaten" for c in MACRO_COLUMNS]
    df[eaten_cols] = df[eaten_cols].ffill().fillna(0.0)

    # Here I forward-fill the capped flag the same way as the macros, so a
    # capped meal's whole "most recent meal" window carries the disclosure
    df["macro_capped"] = np.where(has_meal_row, capped_any.astype(np.float32), np.nan)
    df["macro_capped"] = df["macro_capped"].ffill().fillna(0.0)

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
    df["time_since_last_meal_min"] = df["time_since_last_meal_min"].fillna(24 * 60.0)
    df["has_had_meal"] = has_had_meal

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"cgmacros_{pid:03d}_diet_features.csv")

    desired_cols = ["Timestamp", "Libre GL", "Dexcom GL", "HR", "Calories (Activity)", "METs",
                     "time_since_last_meal_min", "has_had_meal", "macro_capped"] + eaten_cols
    keep_cols = [c for c in desired_cols if c in df.columns]
    missing_cols = [c for c in desired_cols if c not in df.columns]
    if missing_cols:
        print(f"[!] Subject {pid}: missing columns {missing_cols} -- saving without them. "
              f"FLAG as a disclosed per-patient schema gap in your report.")

    n_capped_meals = int(capped_any.sum())
    if n_capped_meals:
        print(f"[i] Subject {pid}: {n_capped_meals}/{n_meals} meals had at least one macro capped.")

    df[keep_cols].to_csv(out_path, index=False)
    print(f"[SAVED] {out_path}")
    return out_path, n_capped_meals


if __name__ == "__main__":
    subj_df = pd.read_csv(T2D_SUBJECT_LIST_CSV)
    t2d_subjects = subj_df[subj_df["is_t2d"]]["subject"].tolist()
    excluded = [p for p in t2d_subjects if p in EXCLUDE_SUBJECTS]
    t2d_subjects = [p for p in t2d_subjects if p not in EXCLUDE_SUBJECTS]
    if excluded:
        print(f"[i] Excluding subject(s) {excluded} from the diet track (see EXCLUDE_SUBJECTS docstring note).")
    print(f"[i] Processing {len(t2d_subjects)} T2D subjects: {t2d_subjects}\n")

    succeeded, failed, total_capped_meals = [], [], 0
    for pid in t2d_subjects:
        try:
            result = extract_diet_features(pid)
            if result:
                out_path, n_capped = result
                succeeded.append(pid)
                total_capped_meals += n_capped
            else:
                failed.append(pid)
        except Exception as e:
            print(f"[!] Subject {pid} FAILED with: {e}")
            print(f"    Skipping this patient, continuing with the rest of the batch.")
            failed.append(pid)
            continue

    print(f"\n{'='*60}")
    print(f"BATCH SUMMARY: {len(succeeded)}/{len(t2d_subjects)} succeeded "
          f"({len(excluded)} excluded, {total_capped_meals} meals capped across the cohort)")
    print(f"  Succeeded: {succeeded}")
    if failed:
        print(f"  Failed entirely: {failed}")
    print(f"{'='*60}")