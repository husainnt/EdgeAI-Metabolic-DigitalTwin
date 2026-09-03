"""
Batch Cohort Data Extraction
============================
Generalizes extract_patient_1205_data.py + extract_patient_1205_sleep.py
across every patient in the HR+CGM+Sleep cohort (577 patients from
cohort_completeness_report.csv), producing one merged CSV per patient:
    results/cohort/patient_<pid>_merged.csv
with columns: timestamp, glucose_mg_dl, heart_rate, sleep_status

Nothing here is guessed -- same JSON schema, same merge_asof logic, same
sleep-interval mapping as the scripts already verified against real files.
Sleep interval matching is vectorized (searchsorted) instead of the
nested-loop version, since that's O(n) per patient vs O(n*m) -- necessary
to make 577 patients tractable (nested loop would take hours).
"""

import os
import json
import glob
import pandas as pd
import numpy as np

DATASET_BASE = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset"
HR_BASE_DIR = os.path.join(DATASET_BASE, "wearable_activity_monitor", "heart_rate", "garmin_vivosmart5")
SLEEP_BASE_DIR = os.path.join(DATASET_BASE, "wearable_activity_monitor", "sleep", "garmin_vivosmart5")
CGM_DIR = os.path.join(DATASET_BASE, "wearable_blood_glucose")

COMPLETENESS_REPORT = "cohort_completeness_report.csv"
OUTPUT_DIR = "results/cohort"


def find_cgm_file(pid):
    for root, _, files in os.walk(CGM_DIR):
        if os.path.basename(root) == pid:
            for f in files:
                if f.endswith(".json"):
                    return os.path.join(root, f)
    return None


def extract_timestamp(entry):
    if not isinstance(entry, dict):
        return None
    eff = entry.get("effective_time_frame", {})
    if isinstance(eff, dict):
        for k in ["date_time", "date_time_start", "start_time"]:
            if k in eff and eff[k]:
                return eff[k]
        ti = eff.get("time_interval", {})
        if isinstance(ti, dict):
            for k in ["start_date_time", "date_time", "start_time"]:
                if k in ti and ti[k]:
                    return ti[k]
    for k in ["date_time", "system_time", "timestamp", "time"]:
        if k in entry and entry[k]:
            return entry[k]
    return None


def extract_glucose_val(entry):
    if not isinstance(entry, dict):
        return None
    bg = entry.get("blood_glucose", {})
    if isinstance(bg, dict) and "value" in bg:
        return bg["value"]
    if "value" in entry:
        return entry["value"]
    return None


def extract_cgm(pid):
    cgm_path = find_cgm_file(pid)
    if not cgm_path:
        return None

    with open(cgm_path, "r") as f:
        raw = json.load(f)

    cgm_entries = []
    if isinstance(raw, dict):
        cgm_entries = raw.get("body", {}).get("cgm", []) or raw.get("body", {}).get("cgm_readings", [])
    elif isinstance(raw, list):
        cgm_entries = raw

    records = []
    for entry in cgm_entries:
        ts = extract_timestamp(entry)
        val = extract_glucose_val(entry)
        if ts is None or val is None:
            continue
        try:
            val_f = float(val)
        except (ValueError, TypeError):
            # Dexcom G6 reports "High" (>400 mg/dL) / "Low" (<40 mg/dL)
            # instead of a number when out of measurable range -- skip these.
            continue
        if val_f > 0:
            records.append({"timestamp": pd.to_datetime(ts), "glucose_mg_dl": val_f})

    if not records:
        return None

    return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)


def extract_hr(pid):
    pid_dir = os.path.join(HR_BASE_DIR, pid)
    if not os.path.isdir(pid_dir):
        return None

    json_files = [f for f in os.listdir(pid_dir) if f.endswith(".json")]
    if not json_files:
        return None

    with open(os.path.join(pid_dir, json_files[0]), "r") as f:
        raw = json.load(f)

    hr_entries = raw.get("body", {}).get("heart_rate", [])

    records = []
    for entry in hr_entries:
        ts = extract_timestamp(entry)
        val = None
        if "heart_rate" in entry and isinstance(entry["heart_rate"], dict):
            val = entry["heart_rate"].get("value")
        elif "value" in entry:
            val = entry.get("value")
        if ts is None or val is None:
            continue
        try:
            val_f = float(val)
        except (ValueError, TypeError):
            continue
        if val_f > 0:
            records.append({"timestamp": pd.to_datetime(ts), "heart_rate": val_f})

    if not records:
        return None

    return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)


def extract_sleep_intervals(pid):
    pid_dir = os.path.join(SLEEP_BASE_DIR, pid)
    pattern = os.path.join(pid_dir, "*.json")
    sleep_files = glob.glob(pattern)
    if not sleep_files:
        return []

    intervals = []
    for file_path in sleep_files:
        with open(file_path, "r") as f:
            data = json.load(f)
        stage_entries = data.get("body", {}).get("sleep", [])
        for entry in stage_entries:
            tf = entry.get("effective_time_frame", {}).get("time_interval", {})
            start_str = tf.get("start_date_time")
            end_str = tf.get("end_date_time")
            if start_str and end_str:
                st = pd.to_datetime(start_str).tz_localize(None)
                et = pd.to_datetime(end_str).tz_localize(None)
                intervals.append((st, et))

    return sorted(intervals, key=lambda x: x[0])


def compute_sleep_flags_vectorized(timestamps, intervals):
    """
    Vectorized sleep-flag mapping using searchsorted instead of a nested
    loop -- necessary to make 577 patients tractable (nested loop is
    O(n_timestamps * n_intervals), which would take far too long at scale).
    Assumes intervals are sorted and non-overlapping (true for nightly sleep).
    """
    if not intervals:
        return np.zeros(len(timestamps), dtype=np.float32)

    starts = np.array([pd.Timestamp(s).value for s, _ in intervals])
    ends = np.array([pd.Timestamp(e).value for _, e in intervals])
    ts_vals = np.array([pd.Timestamp(t).value for t in timestamps])

    idx = np.searchsorted(starts, ts_vals, side="right") - 1
    idx = np.clip(idx, 0, len(starts) - 1)

    flags = np.zeros(len(timestamps), dtype=np.float32)
    valid = idx >= 0
    flags[valid] = (ts_vals[valid] <= ends[idx[valid]]).astype(np.float32)
    return flags


def process_patient(pid):
    df_cgm = extract_cgm(pid)
    df_hr = extract_hr(pid)
    if df_cgm is None or df_hr is None:
        return None, "missing_cgm_or_hr"

    df_merged = pd.merge_asof(
        df_cgm, df_hr, on="timestamp",
        direction="nearest", tolerance=pd.Timedelta(minutes=5)
    )
    df_merged["heart_rate"] = df_merged["heart_rate"].ffill().bfill()

    if df_merged["heart_rate"].isna().all():
        return None, "hr_merge_failed"

    intervals = extract_sleep_intervals(pid)
    if not intervals:
        return None, "missing_sleep"

    timestamps_naive = pd.to_datetime(df_merged["timestamp"]).dt.tz_localize(None)
    sleep_flags = compute_sleep_flags_vectorized(timestamps_naive, intervals)
    df_merged["sleep_status"] = sleep_flags

    if len(df_merged) < 500:  # too short to be useful (< ~1.7 days)
        return None, "too_short"

    return df_merged, "ok"


def run_batch_extraction():
    if not os.path.exists(COMPLETENESS_REPORT):
        raise FileNotFoundError(f"Run build_cohort_completeness_report.py first to create {COMPLETENESS_REPORT}")

    df_report = pd.read_csv(COMPLETENESS_REPORT)

    def to_bool(v):
        return str(v).strip().lower() in ("true", "1")

    cohort_mask = (
        df_report["hr_present"].apply(to_bool)
        & df_report["cgm_present"].apply(to_bool)
        & df_report["sleep_present"].apply(to_bool)
    )
    cohort_ids = df_report.loc[cohort_mask, "person_id"].astype(str).tolist()

    print(f"[+] {len(cohort_ids)} patients in HR+CGM+Sleep cohort to extract.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ok_count = 0
    skip_reasons = {}

    for i, pid in enumerate(cohort_ids):
        out_path = os.path.join(OUTPUT_DIR, f"patient_{pid}_merged.csv")
        if os.path.exists(out_path):
            ok_count += 1
            continue  # already extracted, skip (safe to re-run this script)

        df_merged, status = process_patient(pid)
        if status == "ok":
            df_merged.to_csv(out_path, index=False)
            ok_count += 1
        else:
            skip_reasons[status] = skip_reasons.get(status, 0) + 1

        if (i + 1) % 50 == 0:
            print(f"    ...processed {i + 1}/{len(cohort_ids)} (ok so far: {ok_count})")

    print("\n" + "=" * 60)
    print("BATCH EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Successfully extracted: {ok_count}/{len(cohort_ids)}")
    for reason, count in skip_reasons.items():
        print(f"Skipped ({reason}): {count}")
    print(f"Output directory: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    run_batch_extraction()