"""
Script to extract Garmin step count data for Patient 1031 from OpenmHealth JSON 
and align/resample it to a 5-minute CGM time-series grid.
"""

import os
import json
import pandas as pd
import numpy as np

ACTIVITY_FILE = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset\wearable_activity_monitor\physical_activity\garmin_vivosmart5\1031\1031_activity.json"


def parse_garmin_activity(json_path):
    """
    Parses OpenmHealth formatted activity JSON for Patient 1031.
    Extracts timestamps and step counts from base_movement_quantity.
    """
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
            # Extract timestamp from effective_time_frame
            time_interval = item.get("effective_time_frame", {}).get("time_interval", {})
            time_str = time_interval.get("start_date_time") or item.get("effective_time_frame", {}).get("date_time")

            # Extract step count from base_movement_quantity
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


def resample_steps_to_grid(df, freq="5min"):
    """
    Resamples step counts across fixed intervals (5-min windows).
    Uses sum aggregation to preserve total step volume per window.
    """
    df_resampled = df.set_index("timestamp").resample(freq).agg({
        "steps": "sum"
    }).fillna(0).reset_index()
    return df_resampled


if __name__ == "__main__":
    print(f"Loading and parsing activity data from: {ACTIVITY_FILE}\n")
    steps_df = parse_garmin_activity(ACTIVITY_FILE)

    print(f"[+] Successfully extracted {len(steps_df):,} raw activity records.")
    print(f"[+] Total step count across dataset: {steps_df['steps'].sum():,.0f} steps")

    print("\n--- Non-Zero Step Samples ---")
    active_samples = steps_df[steps_df["steps"] > 0]
    print(active_samples.head(10))

    # Resample to 5-minute grid
    resampled_df = resample_steps_to_grid(steps_df, freq="5min")
    print(f"\n[+] Resampled into {len(resampled_df):,} 5-minute windows.")
    print(resampled_df[resampled_df["steps"] > 0].head(10))