"""
Locates CGM files and inspects dataset folder structure in AI-READI.
"""

import os

DATASET_BASE = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset"

print(f"[+] Checking contents of DATASET_BASE: {DATASET_BASE}")
if os.path.exists(DATASET_BASE):
    top_items = os.listdir(DATASET_BASE)
    print("    Top-level folders/files inside dataset:")
    for item in top_items:
        print(f"    - {item}")
else:
    print(f"[!] Path does not exist: {DATASET_BASE}")

print("\n[+] Searching for Patient 1031 CGM file to identify correct path structure...")
found_files = []
for root, dirs, files in os.walk(DATASET_BASE):
    for f in files:
        if "1031" in f and ("cgm" in f.lower() or "glucose" in f.lower() or f.endswith(".json")):
            found_files.append(os.path.join(root, f))
            if len(found_files) >= 5:
                break
    if len(found_files) >= 5:
        break

if found_files:
    print("    Found matching 1031 files:")
    for path in found_files:
        print(f"    - {path}")
else:
    print("    [!] Could not find any files matching '1031' in DATASET_BASE.")