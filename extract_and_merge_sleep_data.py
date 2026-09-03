"""
Extracts real Garmin sleep stage telemetry (OMH sleep-stages v2.0)
from confirmed F:\ drive directory and merges it into Patient 1031's 
5-minute CGM + HR + Steps dataset.
"""

import os
import glob
import json
import pandas as pd
import numpy as np

PATIENT_1031_BASE_CSV = "results/patient_1031_real_cgm_hr_steps.csv"
OUTPUT_MERGED_CSV = "results/patient_1031_real_cgm_hr_steps_sleep.csv"

# Confirmed AI-READI sleep directory for Patient 1031
SLEEP_DATA_DIR = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset\wearable_activity_monitor\sleep\garmin_vivosmart5\1031"


def load_sleep_intervals():
    pattern = os.path.join(SLEEP_DATA_DIR, "*.json")
    sleep_files = glob.glob(pattern)
    print(f"[i] Searching {SLEEP_DATA_DIR}...")
    print(f"[i] Found {len(sleep_files)} sleep JSON files.")
    
    sleep_intervals = []

    for file_path in sleep_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # OMH sleep-stages v2.0 schema: entries live in body -> sleep
            stage_entries = data.get('body', {}).get('sleep', [])
            
            for entry in stage_entries:
                stage = entry.get('sleep_stage_state')
                tf = entry.get('effective_time_frame', {}).get('time_interval', {})
                start_str = tf.get('start_date_time')
                end_str = tf.get('end_date_time')

                if start_str and end_str:
                    st = pd.to_datetime(start_str).tz_localize(None)
                    et = pd.to_datetime(end_str).tz_localize(None)
                    sleep_intervals.append((st, et, stage))

        except Exception as e:
            print(f"[!] Error reading {file_path}: {e}")
            continue

    print(f"[✓] Extracted {len(sleep_intervals)} real sleep-stage intervals.")
    return sleep_intervals


def merge_sleep_to_patient_csv():
    if not os.path.exists(PATIENT_1031_BASE_CSV):
        print(f"[!] Base CSV missing: {PATIENT_1031_BASE_CSV}")
        return

    df_base = pd.read_csv(PATIENT_1031_BASE_CSV)
    timestamps = pd.to_datetime(df_base['timestamp']).dt.tz_localize(None)
    
    sleep_intervals = load_sleep_intervals()

    if not sleep_intervals:
        raise RuntimeError(
            f"[!] CRITICAL FAILURE: Zero sleep intervals extracted from {SLEEP_DATA_DIR}.\n"
            "Check schema parsing logic."
        )

    sleep_flag = np.zeros(len(df_base), dtype=np.float32)

    # Map 5-minute grid timestamps to active sleep stage intervals
    for idx, ts in enumerate(timestamps):
        for start_t, end_t, stage in sleep_intervals:
            if start_t <= ts <= end_t:
                sleep_flag[idx] = 1.0  # Binary active sleep indicator
                break

    total_hours = (sleep_flag.sum() * 5.0) / 60.0
    print(f"[✓] Mapped {int(sleep_flag.sum())} active 5-minute sleep windows ({total_hours:.1f} total hours across 9.9 days).")

    df_base['sleep_status'] = sleep_flag
    df_base.to_csv(OUTPUT_MERGED_CSV, index=False)
    print(f"[SAVED] Merged dataset saved to: {OUTPUT_MERGED_CSV}")


if __name__ == "__main__":
    merge_sleep_to_patient_csv()