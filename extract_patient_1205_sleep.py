"""
Extracts real Garmin sleep-stage telemetry for Patient 1205 and merges it
into the existing CGM + HR dataset for that patient.
"""

import os
import glob
import json
import pandas as pd
import numpy as np

PATIENT_ID = "1205"
BASE_CSV = f"results/patient_{PATIENT_ID}_real_cgm_hr.csv"
OUTPUT_CSV = f"results/patient_{PATIENT_ID}_real_cgm_hr_sleep.csv"

SLEEP_DATA_DIR = rf"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset\wearable_activity_monitor\sleep\garmin_vivosmart5\{PATIENT_ID}"


def load_sleep_intervals():
    pattern = os.path.join(SLEEP_DATA_DIR, "*.json")
    sleep_files = glob.glob(pattern)
    print(f"[i] Found {len(sleep_files)} sleep JSON files in {SLEEP_DATA_DIR}")

    sleep_intervals = []
    for file_path in sleep_files:
        with open(file_path, "r") as f:
            data = json.load(f)

        # Real OMH sleep-stages v2.0 schema: entries live in body.sleep
        stage_entries = data.get("body", {}).get("sleep", [])
        for entry in stage_entries:
            tf = entry.get("effective_time_frame", {}).get("time_interval", {})
            start_str = tf.get("start_date_time")
            end_str = tf.get("end_date_time")
            if start_str and end_str:
                st = pd.to_datetime(start_str).tz_localize(None)
                et = pd.to_datetime(end_str).tz_localize(None)
                sleep_intervals.append((st, et))

    print(f"[\u2713] Extracted {len(sleep_intervals)} real sleep-stage intervals.")
    return sleep_intervals


def merge_sleep_to_patient_csv():
    if not os.path.exists(BASE_CSV):
        print(f"[!] Base CSV missing: {BASE_CSV}")
        return

    df_base = pd.read_csv(BASE_CSV)
    timestamps = pd.to_datetime(df_base["timestamp"]).dt.tz_localize(None)

    sleep_intervals = load_sleep_intervals()
    if not sleep_intervals:
        raise RuntimeError(
            f"[!] CRITICAL FAILURE: Zero sleep intervals extracted from {SLEEP_DATA_DIR}.\n"
            "Check that Patient 1205 actually has a sleep folder before proceeding."
        )

    sleep_flag = np.zeros(len(df_base), dtype=np.float32)
    for idx, ts in enumerate(timestamps):
        for start_t, end_t in sleep_intervals:
            if start_t <= ts <= end_t:
                sleep_flag[idx] = 1.0
                break

    total_hours = (sleep_flag.sum() * 5.0) / 60.0
    total_span_hours = len(df_base) * 5.0 / 60.0
    print(f"[\u2713] Mapped {int(sleep_flag.sum())} active 5-minute sleep windows "
          f"({total_hours:.1f} hours / {total_span_hours:.1f} total hours).")

    df_base["sleep_status"] = sleep_flag
    df_base.to_csv(OUTPUT_CSV, index=False)
    print(f"[SAVED] Merged dataset saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    merge_sleep_to_patient_csv()