import os
import json
import pandas as pd

# absolute dataset directory configurations
base_dataset_dir = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset"
manifest_path = os.path.join(base_dataset_dir, "wearable_blood_glucose", "manifest.tsv")

print("=== OPEN mHEALTH DEEP SCHEMA INSPECTION ===")

# load manifest data framework
manifest_df = pd.read_csv(manifest_path, sep="\t")
patient_row = manifest_df[manifest_df["person_id"] == 1027]

# resolve target absolute json filepath
relative_path = patient_row["glucose_filepath"].values[0].lstrip("/")
full_path = os.path.join(base_dataset_dir, relative_path.replace("/", os.sep))

if os.path.exists(full_path):
    with open(full_path, "r") as f:
        data = json.load(f)
        
    print("\n=== TOP LEVEL CONTAINER STRUCTURE ===")
    if isinstance(data, dict):
        print(f"Container Type: Dictionary | Top-level Keys: {list(data.keys())}")
        
        print("\n=== ENVELOPE HEADER SNAPSHOT ===")
        print(json.dumps(data.get("header", {}), indent=2)[:400])
        
        print("\n=== TELEMETRY BODY PAYLOAD ANALYSIS ===")
        body = data.get("body", {})
        if isinstance(body, dict):
            print(f"Body Key-Value Schema: {list(body.keys())}")
            for key, val in body.items():
                print(f"\n Inspecting Key '{key}' (Data Type: {type(val)}):")
                print(json.dumps(val, indent=2)[:800])
                print("... [Truncated for readability] ...")
        elif isinstance(body, list):
            print(f"Body is directly formatted as a List array of length: {len(body)}")
            print(json.dumps(body[:2], indent=2))
    else:
        print(f"Container Type: Flat List | Total Element Count: {len(data)}")
        print(json.dumps(data[:2], indent=2))
else:
    print(f"[ERROR] Could not locate file at target path: {full_path}")