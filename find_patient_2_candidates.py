"""
Scans the local AI-READI dataset directory to find ideal candidate IDs for Patient 2 validation.

Filters:
1. Complete Garmin HR and Dexcom CGM data inside wearable_blood_glucose.
2. Excludes Patient 1031 (already tested) and Patient 1027 (severe outlier).
3. Moderate glycemic control (Mean Glucose: 110 - 220 mg/dL).
4. Data span >= 4 days (1,100+ CGM readings).
"""

import os
import json
import pandas as pd
import numpy as np

DATASET_BASE = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset"
HR_DIR = os.path.join(DATASET_BASE, "wearable_activity_monitor", "heart_rate", "garmin_vivosmart5")
CGM_DIR = os.path.join(DATASET_BASE, "wearable_blood_glucose")

EXCLUDE_IDS = {"1031", "1027"}


def find_cgm_file_for_patient(pid):
    """Walks the wearable_blood_glucose directory to find the CGM JSON file for a given patient ID."""
    for root, _, files in os.walk(CGM_DIR):
        if os.path.basename(root) == pid:
            for f in files:
                if f.endswith(".json"):
                    return os.path.join(root, f)
    return None


def parse_cgm_values(cgm_path):
    """Extracts blood glucose values from OMH schema CGM JSON."""
    with open(cgm_path, "r") as f:
        raw = json.load(f)

    cgm_entries = []
    if isinstance(raw, dict):
        cgm_entries = raw.get("body", {}).get("cgm", []) or raw.get("body", {}).get("cgm_readings", [])
    elif isinstance(raw, list):
        cgm_entries = raw

    glucose_vals = []
    for entry in cgm_entries:
        if not isinstance(entry, dict):
            continue

        val = None
        if "blood_glucose" in entry and isinstance(entry["blood_glucose"], dict):
            val = entry["blood_glucose"].get("value")
        elif "value" in entry:
            val = entry.get("value")

        if val and val > 0:
            glucose_vals.append(val)

    return glucose_vals


def find_candidates():
    if not os.path.exists(HR_DIR):
        print(f"[!] Garmin HR directory not found at: {HR_DIR}")
        return
    if not os.path.exists(CGM_DIR):
        print(f"[!] CGM directory not found at: {CGM_DIR}")
        return

    hr_pids = [
        pid for pid in os.listdir(HR_DIR)
        if os.path.isdir(os.path.join(HR_DIR, pid)) and pid not in EXCLUDE_IDS
    ]
    print(f"[+] Found {len(hr_pids):,} patient folders in Garmin HR directory.")

    candidates = []

    for pid in hr_pids:
        pid_hr_dir = os.path.join(HR_DIR, pid)
        json_files = [f for f in os.listdir(pid_hr_dir) if f.endswith(".json")]
        if not json_files:
            continue

        hr_json_path = os.path.join(pid_hr_dir, json_files[0])

        try:
            with open(hr_json_path, "r") as f:
                hr_data = json.load(f)
            hr_entries = hr_data.get("body", {}).get("heart_rate", [])
            if len(hr_entries) < 1000:
                continue
        except Exception:
            continue

        cgm_file = find_cgm_file_for_patient(pid)
        if not cgm_file:
            continue

        try:
            glucose_vals = parse_cgm_values(cgm_file)
            if len(glucose_vals) < 1100:  # ~4+ days of 5-min CGM readings
                continue

            mean_g = float(np.mean(glucose_vals))
            span_days = len(glucose_vals) * 5 / 1440.0

            if 110.0 <= mean_g <= 220.0:
                candidates.append({
                    "patient_id": pid,
                    "cgm_count": len(glucose_vals),
                    "hr_count": len(hr_entries),
                    "span_days": round(span_days, 1),
                    "mean_glucose": round(mean_g, 1)
                })
        except Exception:
            continue

    df_cand = pd.DataFrame(candidates)
    if df_cand.empty:
        print("[!] No candidate patients matched all filtering criteria.")
        return

    df_cand = df_cand.sort_values(by=["cgm_count", "hr_count"], ascending=False).reset_index(drop=True)

    print("\n" + "=" * 65)
    print("      TOP RECOMMENDED CANDIDATES FOR PATIENT 2 VALIDATION     ")
    print("=" * 65)
    print(df_cand.head(10).to_string(index=False))
    print("=" * 65)


if __name__ == "__main__":
    find_candidates()