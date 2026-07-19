import os
import json
import pandas as pd

# here i define the local path to the root dataset folder
dataset_path = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset"
bg_manifest_path = os.path.join(dataset_path, "wearable_blood_glucose", "manifest.tsv")

print("=== TASK 6: CANONICAL PATIENT 1027 INITIALIZATION ===")

# here i read the blood glucose manifest index
manifest_df = pd.read_csv(bg_manifest_path, sep="\t")
patient_bg_info = manifest_df[manifest_df["person_id"] == 1027]

if not patient_bg_info.empty:
    # here i resolve the absolute file path for the cgm file
    rel_path = patient_bg_info["glucose_filepath"].values[0].lstrip("/").replace("/", os.sep)
    abs_bg_path = os.path.join(dataset_path, rel_path)
    
    if os.path.exists(abs_bg_path):
        # here i open and parse the raw json data structure
        with open(abs_bg_path, "r") as f:
            raw_json = json.load(f)
            
        print(f"[SUCCESS] Loaded raw glucose file. Parsing records...")
        
        # here i normalize the nested json arrays into a structured pandas dataframe
        # note: checking for common bids keys or a flat list configuration
        if isinstance(raw_json, dict) and "data" in raw_json:
            cgm_df = pd.DataFrame(raw_json["data"])
        elif isinstance(raw_json, list):
            cgm_df = pd.DataFrame(raw_json)
        else:
            # fallback if the data is keyed directly under patient metadata attributes
            cgm_df = pd.DataFrame(raw_json)
            
        print("\n=== Raw Dataframe Columns Found ===")
        print(cgm_df.columns.tolist())
        print(f"Total rows extracted: {len(cgm_df)}")
        print("\n=== Data Preview ===")
        print(cgm_df.head(5))
        
        # --- VERIFYING COMPANION WEARABLE PATHS ---
        print("\n" + "="*50 + "\n=== VERIFYING FUSION MODALITIES CONVENTION ===")
        hr_path = os.path.join(dataset_path, "wearable_activity_monitor", "heart_rate", "garmin_vivosmart5", "1027")
        activity_path = os.path.join(dataset_path, "wearable_activity_monitor", "physical_activity", "garmin_vivosmart5", "1027")
        
        print(f"Heart Rate Subfolder Exists: {os.path.exists(hr_path)}")
        print(f"Physical Activity Subfolder Exists: {os.path.exists(activity_path)}")
        
    else:
        print(f"[ERROR] Glucose data file missing at: {abs_bg_path}")
else:
    print("[ERROR] Patient 1027 row entry missing from glucose manifest index.")