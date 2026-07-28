"""
Script to inspect Garmin activity timestamps for duplicates, perform safe deduplication,
resample step counts to a 5-minute grid, and merge with patient 1031's CGM + HR dataset.
"""

import os
import json
import pandas as pd
import numpy as np

ACTIVITY_FILE = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset\wearable_activity_monitor\physical_activity\garmin_vivosmart5\1031\1031_activity.json"
CGM_HR_FILE = "results/patient_1031_real_cgm_with_hr.csv"


def parse_garmin_activity(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    body = data.get("body", {})
    activities = body.get("activity", [])

    records = []
    for item in activities:
        if not isinstance(item, dict):
            continue

        try:
            time_interval = item.get("effective_time_frame", {}).get("time_interval", {})
            time_str = time_interval.get("start_date_time") or item.get("effective_time_frame", {}).get("date_time")
            movement = item.get("base_movement_quantity", {})
            step_val = movement.get("value", 0)

            if time_str:
                records.append({
                    "timestamp": pd.to_datetime(time_str),
                    "steps": float(step_val),
                    "activity_type": item.get("activity_name", "unknown")
                })
        except Exception:
            continue

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def analyze_and_deduplicate(df):
    total_raw = len(df)
    duplicate_mask = df.duplicated(subset=["timestamp"], keep=False)
    num_duplicates = duplicate_mask.sum()
    
    print(f"[i] Raw activity records: {total_raw:,}")
    print(f"[i] Records sharing exact duplicate timestamps: {num_duplicates:,}")

    if num_duplicates > 0:
        df_clean = df.groupby("timestamp", as_index=False).agg({
            "steps": "max",
            "activity_type": "first"
        }).sort_values("timestamp").reset_index(drop=True)

        print(f"[+] Records after deduplication: {len(df_clean):,}")
        print(f"[+] Step total change: {df['steps'].sum():,.0f} -> {df_clean['steps'].sum():,.0f} steps")
        return df_clean
    else:
        print("[+] No duplicate timestamps detected.")
        return df


def resample_steps_to_grid(df, freq="5min"):
    # Strip timezone for clean join with CGM timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    
    df_resampled = df.set_index("timestamp").resample(freq).agg({
        "steps": "sum"
    }).fillna(0).reset_index()
    return df_resampled


def merge_steps_with_cgm_hr(cgm_hr_path, resampled_steps):
    if not os.path.exists(cgm_hr_path):
        print(f"[!] Target CGM+HR file not found at: {cgm_hr_path}")
        return None

    cgm_df = pd.read_csv(cgm_hr_path)
    
    # Standardize timestamps to naive datetimes on both ends
    cgm_df['timestamp'] = pd.to_datetime(cgm_df['timestamp']).dt.tz_localize(None)
    resampled_steps['timestamp'] = pd.to_datetime(resampled_steps['timestamp']).dt.tz_localize(None)

    # Use merge_asof with nearest 5-minute tolerance to guarantee match
    merged_df = pd.merge_asof(
        cgm_df.sort_values('timestamp'),
        resampled_steps.sort_values('timestamp'),
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('5min')
    )
    merged_df['steps'] = merged_df['steps'].fillna(0.0)

    return merged_df


if __name__ == "__main__":
    print(f"Parsing activity data from: {ACTIVITY_FILE}\n")
    raw_steps = parse_garmin_activity(ACTIVITY_FILE)

    clean_steps = analyze_and_deduplicate(raw_steps)
    resampled_steps = resample_steps_to_grid(clean_steps, freq="5min")
    merged_dataset = merge_steps_with_cgm_hr(CGM_HR_FILE, resampled_steps)

    if merged_dataset is not None:
        out_path = "results/patient_1031_real_cgm_hr_steps.csv"
        os.makedirs("results", exist_ok=True)
        merged_dataset.to_csv(out_path, index=False)
        print(f"\n[+] Successfully created merged dataset: {out_path}")
        print(f"    Total rows: {len(merged_dataset):,}")
        print(f"    Columns: {list(merged_dataset.columns)}")
        print(f"    Matched non-zero step rows: {(merged_dataset['steps'] > 0).sum():,}")
        print("\n--- Sample Active Rows ---")
        print(merged_dataset[merged_dataset['steps'] > 0].head())