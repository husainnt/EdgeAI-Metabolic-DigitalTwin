"""
Inspects the raw schema of Garmin Sleep JSON files for Patient 1031.
"""

import glob
import json
import os

SLEEP_DIR = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset\wearable_activity_monitor\sleep\garmin_vivosmart5\1031"

files = glob.glob(os.path.join(SLEEP_DIR, "*.json"))
print(f"[i] Found {len(files)} sleep JSON files in {SLEEP_DIR}")

if files:
    with open(files[0], 'r') as f:
        sample_data = json.load(f)
    print("\n--- SAMPLE SLEEP JSON SCHEMA (First 1000 Chars) ---")
    print(json.dumps(sample_data, indent=2)[:1000])
else:
    print("[!] No JSON files found in path! Check drive letter or directory structure.")