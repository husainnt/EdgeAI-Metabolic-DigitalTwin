"""
Diagnostic: inspect the real structure of one physical_activity JSON file
so we can fix the body_key mismatch in build_cohort_completeness_report.py
"""
import os
import json

DATASET_BASE = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset"
ACTIVITY_DIR = os.path.join(DATASET_BASE, "wearable_activity_monitor", "physical_activity", "garmin_vivosmart5")

if not os.path.isdir(ACTIVITY_DIR):
    print(f"[!] ACTIVITY_DIR does not exist: {ACTIVITY_DIR}")
    parent = os.path.join(DATASET_BASE, "wearable_activity_monitor")
    if os.path.isdir(parent):
        print("[!] Listing wearable_activity_monitor/ subfolders instead:")
        print(os.listdir(parent))
else:
    pids = [p for p in os.listdir(ACTIVITY_DIR) if os.path.isdir(os.path.join(ACTIVITY_DIR, p))]
    print(f"[+] Found {len(pids)} patient folders under physical_activity")
    if pids:
        sample_pid_dir = os.path.join(ACTIVITY_DIR, pids[0])
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
                else:
                    print(f"    body['{k}'] = {v}")