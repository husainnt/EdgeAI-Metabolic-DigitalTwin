"""
Batch SpO2 + Activity extraction for the full oral-med T2D cohort.

Generalizes extract_and_merge_spo2.py and extract_and_merge_activity.py
(originally hardcoded to Patient 1031) across every patient in
results/cohort/ that already has an HR+CGM+Sleep merged file -- i.e. the
577-patient set from cohort_completeness_report.csv, of which ~444 are
expected to actually have SpO2+Activity data (per the completeness report
already run this project).

For each patient:
    1. Merge SpO2 (merge_asof nearest, 10-min tolerance) -- same logic as
       extract_and_merge_spo2.py
    2. Merge Activity (searchsorted 5-min binning: steps_sum, walking_frac)
       -- same logic as extract_and_merge_activity.py
    3. Overwrite that patient's results/cohort/patient_<id>_merged.csv
       in-place with the two new columns added (glucose/HR/sleep columns
       untouched)

This does NOT touch patients that have no SpO2/Activity folder at all --
they're left with just HR+Sleep+Calib, honestly reflecting real data
availability, same principle as every other has_x flag in this project.

Usage:
    python batch_extract_spo2_activity.py
"""

import os
import glob
import json
import numpy as np
import pandas as pd

DATASET_BASE = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset"
SPO2_DIR = os.path.join(DATASET_BASE, "wearable_activity_monitor", "oxygen_saturation", "garmin_vivosmart5")
ACTIVITY_DIR = os.path.join(DATASET_BASE, "wearable_activity_monitor", "physical_activity", "garmin_vivosmart5")

COHORT_DIR = "results/cohort"
MATCH_TOLERANCE_MINUTES = 10


def load_spo2_readings(pid: str):
    pid_dir = os.path.join(SPO2_DIR, pid)
    if not os.path.isdir(pid_dir):
        return []

    records = []
    for jf in glob.glob(os.path.join(pid_dir, "*.json")):
        try:
            with open(jf, "r") as f:
                data = json.load(f)
            for entry in data.get("body", {}).get("breathing", []):
                val = entry.get("oxygen_saturation", {}).get("value")
                dt_str = entry.get("effective_time_frame", {}).get("date_time")
                if val is None or dt_str is None:
                    continue
                records.append((pd.to_datetime(dt_str).tz_localize(None), float(val)))
        except Exception:
            continue
    return records


def load_activity_entries(pid: str):
    pid_dir = os.path.join(ACTIVITY_DIR, pid)
    if not os.path.isdir(pid_dir):
        return []

    records = []
    for jf in glob.glob(os.path.join(pid_dir, "*.json")):
        try:
            with open(jf, "r") as f:
                data = json.load(f)
            for entry in data.get("body", {}).get("activity", []):
                raw_val = entry.get("base_movement_quantity", {}).get("value", "")
                if raw_val == "" or raw_val is None:
                    continue
                try:
                    steps_val = float(raw_val)
                except (TypeError, ValueError):
                    continue
                start_str = entry.get("effective_time_frame", {}).get("time_interval", {}).get("start_date_time")
                if start_str is None:
                    continue
                ts = pd.to_datetime(start_str).tz_localize(None)
                act_name = entry.get("activity_name", "")
                records.append((ts, steps_val, act_name))
        except Exception:
            continue
    return records


def merge_spo2(df_base: pd.DataFrame, spo2_records):
    if not spo2_records:
        return df_base, 0.0

    df_spo2 = pd.DataFrame(spo2_records, columns=["timestamp_naive", "oxygen_saturation"])
    df_spo2 = df_spo2.sort_values("timestamp_naive").reset_index(drop=True)

    df_base = df_base.sort_values("timestamp_naive").reset_index(drop=True)
    merged = pd.merge_asof(
        df_base, df_spo2, on="timestamp_naive",
        direction="nearest", tolerance=pd.Timedelta(minutes=MATCH_TOLERANCE_MINUTES),
    )
    match_pct = 100 * merged["oxygen_saturation"].notna().sum() / len(merged) if len(merged) else 0.0
    return merged, match_pct


def merge_activity(df_base: pd.DataFrame, activity_records):
    if not activity_records:
        df_base["steps_sum"] = 0.0
        df_base["activity_walking_frac"] = 0.0
        return df_base, 0.0

    grid_ts = pd.to_datetime(df_base["timestamp"]).dt.tz_localize(None).values.astype("datetime64[ns]")
    n_grid = len(grid_ts)

    raw_ts = np.array([r[0] for r in activity_records], dtype="datetime64[ns]")
    raw_steps = np.array([r[1] for r in activity_records], dtype=np.float64)
    raw_is_walk = np.array([r[2] == "walking" for r in activity_records], dtype=np.float64)

    bin_idx = np.searchsorted(grid_ts, raw_ts, side="left")
    valid = (bin_idx >= 0) & (bin_idx < n_grid)

    steps_sum = np.zeros(n_grid, dtype=np.float64)
    walk_count = np.zeros(n_grid, dtype=np.float64)
    total_count = np.zeros(n_grid, dtype=np.float64)
    np.add.at(steps_sum, bin_idx[valid], raw_steps[valid])
    np.add.at(walk_count, bin_idx[valid], raw_is_walk[valid])
    np.add.at(total_count, bin_idx[valid], 1.0)

    walking_frac = np.divide(walk_count, total_count, out=np.zeros_like(walk_count), where=total_count > 0)
    coverage_pct = 100 * (total_count > 0).sum() / n_grid if n_grid else 0.0

    df_base["steps_sum"] = steps_sum
    df_base["activity_walking_frac"] = walking_frac
    return df_base, coverage_pct


def process_patient(csv_path: str):
    pid = os.path.basename(csv_path).replace("patient_", "").replace("_merged.csv", "")
    df = pd.read_csv(csv_path)

    if "oxygen_saturation" in df.columns and "steps_sum" in df.columns:
        return pid, "already_done", None, None

    df["timestamp_naive"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    spo2_records = load_spo2_readings(pid)
    df, spo2_match_pct = merge_spo2(df, spo2_records)

    activity_records = load_activity_entries(pid)
    df, activity_coverage_pct = merge_activity(df, activity_records)

    df = df.drop(columns=["timestamp_naive"])
    df.to_csv(csv_path, index=False)

    has_spo2 = len(spo2_records) > 0
    has_activity = len(activity_records) > 0
    status = f"spo2={'Y' if has_spo2 else 'N'}({spo2_match_pct:.0f}%) act={'Y' if has_activity else 'N'}({activity_coverage_pct:.0f}%)"
    return pid, status, spo2_match_pct, activity_coverage_pct


def run_batch():
    csv_files = sorted(glob.glob(os.path.join(COHORT_DIR, "patient_*_merged.csv")))
    print(f"[i] Found {len(csv_files)} cohort patient files to process.\n")

    both_count = 0
    spo2_match_rates = []
    activity_coverage_rates = []

    for i, csv_path in enumerate(csv_files):
        pid, status, spo2_pct, act_pct = process_patient(csv_path)
        if status == "already_done":
            continue
        if spo2_pct is not None:
            spo2_match_rates.append(spo2_pct)
        if act_pct is not None:
            activity_coverage_rates.append(act_pct)
        if spo2_pct and spo2_pct > 0 and act_pct and act_pct > 0:
            both_count += 1

        if (i + 1) % 50 == 0:
            print(f"    ...processed {i + 1}/{len(csv_files)}")

    print(f"\n[+] Done. Patients with BOTH SpO2 and Activity data: {both_count}/{len(csv_files)}")
    if spo2_match_rates:
        print(f"[i] SpO2 match rate across cohort -- mean: {np.mean(spo2_match_rates):.1f}%, "
              f"min: {np.min(spo2_match_rates):.1f}%, max: {np.max(spo2_match_rates):.1f}%")
    if activity_coverage_rates:
        print(f"[i] Activity coverage across cohort -- mean: {np.mean(activity_coverage_rates):.1f}%, "
              f"min: {np.min(activity_coverage_rates):.1f}%, max: {np.max(activity_coverage_rates):.1f}%")


if __name__ == "__main__":
    run_batch()