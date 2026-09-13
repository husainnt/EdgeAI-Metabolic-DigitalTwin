"""
Data-quality inspection: Activity (physical_activity) real value population
check for Patient 1031, before building extract_and_merge_activity.py.

WHY THIS EXISTS: Debug_activity_schema.py showed base_movement_quantity.value
as an EMPTY STRING in the very first entry inspected. Before writing a merge
script that assumes usable step-count data, we need to know what fraction of
Patient 1031's activity entries actually carry a real numeric value vs an
empty placeholder -- the same class of schema trap that bit the
sleep/oxygen_saturation body-key extraction earlier in this project. Run
this FIRST; do not build the real extraction script until you've seen this
output.

Usage:
    python inspect_activity_data_quality.py
"""

import os
import json

DATASET_BASE = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset"
ACTIVITY_DIR = os.path.join(DATASET_BASE, "wearable_activity_monitor", "physical_activity", "garmin_vivosmart5")
PATIENT_ID = "1031"


def inspect_patient_activity(pid: str):
    pid_dir = os.path.join(ACTIVITY_DIR, pid)
    if not os.path.isdir(pid_dir):
        print(f"[!] No activity folder found for patient {pid}: {pid_dir}")
        return

    json_files = [f for f in os.listdir(pid_dir) if f.endswith(".json")]
    if not json_files:
        print(f"[!] No activity JSON files found for patient {pid}")
        return

    total_entries = 0
    numeric_entries = 0
    empty_entries = 0
    non_numeric_nonzero_entries = 0
    value_samples = []
    unit_counts = {}
    activity_name_counts = {}

    for jf in json_files:
        path = os.path.join(pid_dir, jf)
        with open(path, "r") as f:
            data = json.load(f)

        entries = data.get("body", {}).get("activity", [])
        for entry in entries:
            total_entries += 1
            bmq = entry.get("base_movement_quantity", {})
            raw_val = bmq.get("value", "")
            unit = bmq.get("unit", "")
            unit_counts[unit] = unit_counts.get(unit, 0) + 1

            act_name = entry.get("activity_name", "")
            activity_name_counts[act_name] = activity_name_counts.get(act_name, 0) + 1

            if raw_val == "" or raw_val is None:
                empty_entries += 1
                continue

            try:
                numeric_val = float(raw_val)
                numeric_entries += 1
                if len(value_samples) < 10:
                    value_samples.append(numeric_val)
            except (TypeError, ValueError):
                non_numeric_nonzero_entries += 1

    print("=" * 60)
    print(f"ACTIVITY DATA QUALITY REPORT -- Patient {pid}")
    print("=" * 60)
    print(f"Total entries:                  {total_entries:,}")
    if total_entries == 0:
        print("[!] No entries found at all -- nothing further to report.")
        return
    print(f"Empty value ('' or None):       {empty_entries:,} ({100*empty_entries/total_entries:.1f}%)")
    print(f"Valid numeric value:            {numeric_entries:,} ({100*numeric_entries/total_entries:.1f}%)")
    print(f"Non-numeric, non-empty value:   {non_numeric_nonzero_entries:,} ({100*non_numeric_nonzero_entries/total_entries:.1f}%)")
    print(f"\nUnit breakdown: {unit_counts}")
    top_names = sorted(activity_name_counts.items(), key=lambda x: -x[1])[:5]
    print(f"activity_name breakdown (top 5): {top_names}")
    print(f"\nSample numeric values (first 10 found): {value_samples}")
    print("=" * 60)

    frac_usable = numeric_entries / total_entries
    if frac_usable < 0.5:
        print("\n[!] WARNING: fewer than half of entries carry a usable numeric value.")
        print("    Do NOT treat this like continuously-sampled HR data. Consider:")
        print("    - resampling only around timestamps that DO have real values")
        print("    - dropping windows with no real value inside them, rather than")
        print("      filling gaps with 0 (0 steps is a real, different signal from")
        print("      'no reading taken')")
    else:
        print("\n[+] Majority of entries carry a usable numeric value -- safe to build")
        print("    a merge_asof-style extraction similar to extract_and_merge_spo2.py.")


if __name__ == "__main__":
    inspect_patient_activity(PATIENT_ID)