"""
Find and Inspect Patient 1031 Wearable/HR Files on Drive F:
"""

import os
import pandas as pd

f_drive_root = "F:\\"
target_patient = "1031"

print(f"Scanning {f_drive_root} for Patient {target_patient} wearable/HR files...")
print("=" * 70)

matches = []
for root, dirs, files in os.walk(f_drive_root):
    for file in files:
        file_lower = file.lower()
        if target_patient in file and any(k in file_lower for k in ['hr', 'heart', 'fitbit', 'pulse', 'wearable']):
            matches.append(os.path.join(root, file))

if not matches:
    # Broader search if patient ID is in the folder name instead of filename
    print(f"No direct filename match for '{target_patient}'. Searching patient directories...")
    for root, dirs, files in os.walk(f_drive_root):
        if target_patient in root:
            for file in files:
                file_lower = file.lower()
                if any(k in file_lower for k in ['hr', 'heart', 'fitbit', 'pulse', 'wearable', 'csv']):
                    matches.append(os.path.join(root, file))

if matches:
    print(f"✓ Found {len(matches)} potential wearable file(s):\n")
    for m in matches[:5]:  # Show top 5
        print(f"📄 Path: {m}")
        try:
            sample_df = pd.read_csv(m, nrows=5)
            print(f"   Columns: {sample_df.columns.tolist()}")
            print(f"   First timestamp: {sample_df.iloc[0].to_dict()}\n")
        except Exception as e:
            print(f"   (Could not read file preview: {e})\n")
else:
    print("❌ No matching HR files found. Let's list the main folders on F:\\ to locate the AI-READI directory structure.")
    print("Directory listing for F:\\:")
    try:
        print(os.listdir(f_drive_root))
    except Exception as e:
        print(f"Error accessing F:\\: {e}")