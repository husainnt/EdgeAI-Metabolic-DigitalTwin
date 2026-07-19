import os
import json
import pandas as pd

# Absolute dataset directory configurations
base_dataset_dir = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset"
manifest_path = os.path.join(base_dataset_dir, "wearable_blood_glucose", "manifest.tsv")

print("=== TASK 6: PARSING CANONICAL PATIENT 1031 & CLINICAL METRICS ===")

# Load manifest and isolate patient 1031
manifest_df = pd.read_csv(manifest_path, sep="\t")
patient_row = manifest_df[manifest_df["person_id"] == 1031]

# Resolve relative path to absolute file system location
relative_path = patient_row["glucose_filepath"].values[0].lstrip("/")
full_path = os.path.join(base_dataset_dir, relative_path.replace("/", os.sep))

if os.path.exists(full_path):
    with open(full_path, "r") as f:
        data = json.load(f)
        
    cgm_records = data["body"]["cgm"]
    print(f"[SUCCESS] Total raw CGM records found: {len(cgm_records)}")
    
    # Extract timestamps and values from Open mHealth nested schema
    rows = []
    for rec in cgm_records:
        try:
            ts = rec["effective_time_frame"]["time_interval"]["start_date_time"]
            glucose = rec["blood_glucose"]["value"]
            rows.append({"timestamp": ts, "glucose_mg_dl": glucose})
        except KeyError:
            continue
            
    # Build clean DataFrame
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Secure numeric typing by coercing text anomalies into NaN and dropping them
    df["glucose_mg_dl"] = pd.to_numeric(df["glucose_mg_dl"], errors='coerce')
    df = df.dropna(subset=["glucose_mg_dl"])
    
    # Sort chronologically to ensure pristine timeline ordering
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Compute clinical metrics matching synthetic simulation profiles
    mean_glucose = df["glucose_mg_dl"].mean()
    peak = df["glucose_mg_dl"].max()
    minimum = df["glucose_mg_dl"].min()
    tir = ((df["glucose_mg_dl"] >= 70) & (df["glucose_mg_dl"] <= 180)).mean() * 100
    tar = (df["glucose_mg_dl"] > 180).mean() * 100
    tbr = (df["glucose_mg_dl"] < 70).mean() * 100
    
    print("\n=== PATIENT 1031 REAL CGM CLINICAL SUMMARY ===")
    print(f"Mean Glucose:    {mean_glucose:.1f} mg/dL")
    print(f"Peak Glucose:    {peak} mg/dL")
    print(f"Minimum Glucose: {minimum} mg/dL")
    print(f"TIR (70-180):    {tir:.1f}%")
    print(f"TAR (>180):      {tar:.1f}%")
    print(f"TBR (<70):       {tbr:.1f}%")
    print(f"Date range:      {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Export cleaned baseline dataset for direct Layer 2 LSTM supervision
    os.makedirs("results", exist_ok=True)
    output_csv = "results/patient_1031_real_cgm.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n[SAVED] Cleaned time-series array exported to: {output_csv}")
    
else:
    print(f"[ERROR] Could not locate file at target path: {full_path}")