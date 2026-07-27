"""
OpenmHealth / AI-READI Garmin Heart Rate Parser & CGM Merging Script
"""

import json
import os
import numpy as np
import pandas as pd

JSON_PATH = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset\wearable_activity_monitor\heart_rate\garmin_vivosmart5\1031\1031_heartrate.json"
CGM_CSV_PATH = "results/patient_1031_real_cgm.csv"
OUTPUT_CSV_PATH = "results/patient_1031_real_cgm_with_hr.csv"

print("Step 1: Unpacking AI-READI / OpenmHealth Nested Heart Rate JSON...")
print("=" * 70)

if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(f"Could not find JSON file at {JSON_PATH}")

with open(JSON_PATH, "r") as f:
    raw_data = json.load(f)

# Locate the nested record list
hr_list = []
if isinstance(raw_data, dict):
    if "body.heart_rate" in raw_data:
        hr_list = raw_data["body.heart_rate"]
    elif "body" in raw_data and isinstance(raw_data["body"], dict) and "heart_rate" in raw_data["body"]:
        hr_list = raw_data["body"]["heart_rate"]
    elif "body" in raw_data and isinstance(raw_data["body"], list):
        hr_list = raw_data["body"]
elif isinstance(raw_data, list):
    hr_list = raw_data

# Flatten nested list
records = []
for entry in hr_list:
    if not isinstance(entry, dict):
        continue

    # Extract BPM value
    bpm = None
    if "heart_rate" in entry and isinstance(entry["heart_rate"], dict):
        bpm = entry["heart_rate"].get("value")
    elif "value" in entry:
        bpm = entry.get("value")

    # Extract Timestamp
    ts = None
    if "effective_time_frame" in entry and isinstance(entry["effective_time_frame"], dict):
        tf = entry["effective_time_frame"]
        if "date_time" in tf:
            ts = tf["date_time"]
        elif "time_interval" in tf and isinstance(tf["time_interval"], dict):
            ts = tf["time_interval"].get("start_date_time") or tf["time_interval"].get("end_date_time")
    elif "date_time" in entry:
        ts = entry["date_time"]
    elif "header" in entry and isinstance(entry["header"], dict):
        ts = entry["header"].get("creation_date_time")

    if bpm is not None and ts is not None:
        records.append({"timestamp": ts, "heart_rate": bpm})

df_hr = pd.DataFrame(records)
print(f"✓ Successfully extracted {len(df_hr)} heart rate samples from JSON!")

if len(df_hr) == 0:
    print("⚠ Could not automatically parse records. Inspecting first entry structure:")
    print(hr_list[:1])
    exit()

# Clean & Parse Timestamps
df_hr['timestamp'] = pd.to_datetime(df_hr['timestamp'], utc=True).dt.tz_localize(None)
df_hr['heart_rate'] = pd.to_numeric(df_hr['heart_rate'], errors='coerce')

# Filter out zero values or invalid sensor drops (< 30 BPM or > 220 BPM)
df_hr = df_hr[(df_hr['heart_rate'] >= 30) & (df_hr['heart_rate'] <= 220)]
df_hr = df_hr.sort_values('timestamp').reset_index(drop=True)

print(f"✓ Cleaned HR Records: {len(df_hr)} valid readings")
print(f"  • Date Range: {df_hr['timestamp'].min()} to {df_hr['timestamp'].max()}")
print(f"  • Mean HR:    {df_hr['heart_rate'].mean():.1f} BPM")
print(f"  • Min/Max HR:  {df_hr['heart_rate'].min():.1f} / {df_hr['heart_rate'].max():.1f} BPM")

print("\n" + "=" * 70)
print("Step 2: Merging Real Garmin HR with Patient 1031 CGM Timeline...")

# Load CGM Data
df_cgm = pd.read_csv(CGM_CSV_PATH)
time_cols = [c for c in df_cgm.columns if any(k in c.lower() for k in ['time', 'date', 'timestamp'])]
df_cgm['timestamp'] = pd.to_datetime(df_cgm[time_cols[0]], utc=True).dt.tz_localize(None)
df_cgm = df_cgm.sort_values('timestamp').reset_index(drop=True)

# Merge via nearest-timestamp join (asof) within 15 min tolerance
merged_df = pd.merge_asof(
    df_cgm,
    df_hr[['timestamp', 'heart_rate']],
    on='timestamp',
    direction='nearest',
    tolerance=pd.Timedelta(minutes=15)
)

# Check missing count before interpolation
missing_before = merged_df['heart_rate'].isna().sum()
merged_df['heart_rate'] = merged_df['heart_rate'].interpolate(method='linear').bfill().ffill()

print(f"✓ Merge Complete!")
print(f"  • Total CGM Samples:  {len(merged_df)}")
print(f"  • Matched HR Points:  {len(merged_df) - missing_before} / {len(merged_df)}")
print(f"  • Merged Mean HR:     {merged_df['heart_rate'].mean():.1f} BPM")

# Save merged dataset
os.makedirs("results", exist_ok=True)
merged_df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"\n[SAVED] Validated multimodal dataset to: {OUTPUT_CSV_PATH}")