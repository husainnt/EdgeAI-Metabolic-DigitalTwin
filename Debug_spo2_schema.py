"""
Diagnostic: inspect the real structure of one oxygen_saturation JSON file
so we can fix the body_key mismatch in build_cohort_completeness_report.py
"""
import os
import json

DATASET_BASE = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset"
SPO2_DIR = os.path.join(DATASET_BASE, "wearable_activity_monitor", "oxygen_saturation", "garmin_vivosmart5")

if not os.path.isdir(SPO2_DIR):
    print(f"[!] SPO2_DIR does not exist: {SPO2_DIR}")
    print("[!] Listing wearable_activity_monitor/ subfolders instead:")
    parent = os.path.join(DATASET_BASE, "wearable_activity_monitor")
    if os.path.isdir(parent):
        print(os.listdir(parent))
else:
    pids = [p for p in os.listdir(SPO2_DIR) if os.path.isdir(os.path.join(SPO2_DIR, p))]
    print(f"[+] Found {len(pids)} patient folders under oxygen_saturation")
    if pids:
        sample_pid_dir = os.path.join(SPO2_DIR, pids[0])
        json_files = [f for f in os.listdir(sample_pid_dir) if f.endswith(".json")]
        print(f"[+] Sample patient {pids[0]} has files: {json_files}")
        if json_files:
            with open(os.path.join(sample_pid_dir, json_files[0]), "r") as f:
                data = json.load(f)
            print("[+] Top-level keys:", list(data.keys()))
            body = data.get("body", {})
            print("[+] body keys:", list(body.keys()))
            for k, v in body.items():
                if isinstance(v, list):
                    print(f"    body['{k}'] is a list of length {len(v)}")
                    if v:
                        print(f"    first entry: {v[0]}")