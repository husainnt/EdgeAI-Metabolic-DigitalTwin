"""
Extracts and aligns continuous Dexcom G6 CGM and Garmin Vivosmart 5 HR data
for Patient 1205 onto a synchronized 5-minute timestamp grid.
Robust to OMH schema timestamp variations.
"""

import os
import json
import pandas as pd
import numpy as np

DATASET_BASE = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset"
PATIENT_ID = "1205"

HR_DIR = os.path.join(DATASET_BASE, "wearable_activity_monitor", "heart_rate", "garmin_vivosmart5", PATIENT_ID)
CGM_DIR = os.path.join(DATASET_BASE, "wearable_blood_glucose")
OUTPUT_PATH = f"results/patient_{PATIENT_ID}_real_cgm_hr.csv"


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

    # 1. Check nested effective_time_frame
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

    # 2. Check top-level timestamp keys
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


def extract_cgm():
    cgm_path = find_cgm_file(PATIENT_ID)
    if not cgm_path:
        raise FileNotFoundError(f"Could not locate CGM JSON file for Patient {PATIENT_ID} in {CGM_DIR}")

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

        if ts and val is not None and float(val) > 0:
            records.append({"timestamp": pd.to_datetime(ts), "glucose_mg_dl": float(val)})

    if not records:
        print("[!] Warning: Zero CGM records parsed. Sample first entry:")
        print(cgm_entries[0] if cgm_entries else "Empty list")
        raise ValueError("Failed to parse CGM timestamps/values.")

    df_cgm = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    return df_cgm


def extract_hr():
    if not os.path.exists(HR_DIR):
        raise FileNotFoundError(f"HR directory missing: {HR_DIR}")

    json_files = [f for f in os.listdir(HR_DIR) if f.endswith(".json")]
    hr_path = os.path.join(HR_DIR, json_files[0])

    with open(hr_path, "r") as f:
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

        if ts and val is not None and float(val) > 0:
            records.append({"timestamp": pd.to_datetime(ts), "heart_rate": float(val)})

    df_hr = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    return df_hr


def run_extraction():
    print(f"[+] Extracting dataset for Patient {PATIENT_ID}...")
    df_cgm = extract_cgm()
    df_hr = extract_hr()

    print(f"    Raw CGM readings: {len(df_cgm):,} (Range: {df_cgm['timestamp'].min()} to {df_cgm['timestamp'].max()})")
    print(f"    Raw HR readings:  {len(df_hr):,} (Range: {df_hr['timestamp'].min()} to {df_hr['timestamp'].max()})")

    # Align HR onto CGM 5-minute timestamps
    df_merged = pd.merge_asof(
        df_cgm,
        df_hr,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=5)
    )

    df_merged['heart_rate'] = df_merged['heart_rate'].ffill().bfill()

    os.makedirs("results", exist_ok=True)
    df_merged.to_csv(OUTPUT_PATH, index=False)

    print(f"\n[+] Aligned dataset saved to: {OUTPUT_PATH}")
    print(f"    Total 5-min intervals: {len(df_merged):,}")
    print(f"    Mean Glucose:          {df_merged['glucose_mg_dl'].mean():.1f} mg/dL")
    print(f"    Mean Heart Rate:       {df_merged['heart_rate'].mean():.1f} bpm")
    print(df_merged.head())


if __name__ == "__main__":
    run_extraction()