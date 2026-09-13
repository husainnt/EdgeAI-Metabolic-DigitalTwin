"""
Extracts real Garmin SpO2 (oxygen saturation) telemetry (OMH 'breathing'
schema) and merges it onto Patient 1031's existing 5-minute grid dataset.

Schema confirmed via Debug_spo2_schema.py:
    body['breathing'] = [
        {'oxygen_saturation': {'value': <int>, 'unit': '%'},
         'effective_time_frame': {'date_time': <ISO8601>},
         'measurement_method': 'pulse oximetry'},
        ...
    ]

Unlike sleep stages (intervals with start/end), SpO2 readings are
POINT-IN-TIME. This uses pd.merge_asof(direction='nearest', tolerance=...)
rather than the interval-overlap loop used in extract_and_merge_sleep_data.py.

IMPORTANT -- honesty about coverage: not every 5-minute grid timestep will
have a real SpO2 reading within tolerance. Those timesteps are left NaN here
(has_spo2 handling and any fill strategy happens downstream in
window_generator.py) rather than silently defaulting to some fixed value.
The match-rate percentage printed below tells you how much of the dataset
is genuinely covered -- report this number, don't just report "SpO2 wired
in" as binary.

OUTPUT NOTE: this writes to a NEW file rather than overwriting
results/cohort/patient_1031_merged.csv directly, since it's not yet
confirmed how that file relates to the patient_1031_real_cgm_hr_steps_sleep*
chain of files. Inspect the output, confirm columns/match-rate look right,
THEN copy/rename it over whatever path window_generator.py actually reads.

Usage:
    python extract_and_merge_spo2.py
"""

import os
import glob
import json
import pandas as pd

# --- Update these two paths to match whichever file is your current
#     "real, about-to-be-used-for-training" Patient 1031 dataset ---
INPUT_CSV = "results/patient_1031_real_cgm_hr_steps_sleep.csv"
OUTPUT_CSV = "results/patient_1031_real_cgm_hr_steps_sleep_spo2.csv"

SPO2_DATA_DIR = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset\wearable_activity_monitor\oxygen_saturation\garmin_vivosmart5\1031"

# How far (minutes) a real SpO2 reading may drift from a grid timestamp
# before we treat it as "no reading at this timestep" rather than matching
# it across a real gap.
MATCH_TOLERANCE_MINUTES = 10


def load_spo2_readings():
    pattern = os.path.join(SPO2_DATA_DIR, "*.json")
    spo2_files = glob.glob(pattern)
    print(f"[i] Searching {SPO2_DATA_DIR}...")
    print(f"[i] Found {len(spo2_files)} SpO2 JSON files.")

    records = []
    for file_path in spo2_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            entries = data.get("body", {}).get("breathing", [])
            for entry in entries:
                spo2_obj = entry.get("oxygen_saturation", {})
                val = spo2_obj.get("value")
                dt_str = entry.get("effective_time_frame", {}).get("date_time")

                if val is None or dt_str is None:
                    continue

                ts = pd.to_datetime(dt_str).tz_localize(None)
                records.append((ts, float(val)))

        except Exception as e:
            print(f"[!] Error reading {file_path}: {e}")
            continue

    print(f"[+] Extracted {len(records)} real SpO2 readings.")
    return records


def merge_spo2_to_patient_csv():
    if not os.path.exists(INPUT_CSV):
        print(f"[!] Base CSV missing: {INPUT_CSV}")
        print("    Update INPUT_CSV at the top of this script to your actual current file.")
        return

    df_base = pd.read_csv(INPUT_CSV)
    df_base["timestamp_naive"] = pd.to_datetime(df_base["timestamp"]).dt.tz_localize(None)

    records = load_spo2_readings()
    if not records:
        raise RuntimeError(
            f"[!] CRITICAL FAILURE: Zero SpO2 readings extracted from {SPO2_DATA_DIR}.\n"
            "Check schema parsing logic before proceeding -- do not assume this means"
            " 'patient has no SpO2 data' without checking the raw JSON directly."
        )

    df_spo2 = pd.DataFrame(records, columns=["timestamp_naive", "oxygen_saturation"])
    df_spo2 = df_spo2.sort_values("timestamp_naive").reset_index(drop=True)
    df_base_sorted = df_base.sort_values("timestamp_naive").reset_index(drop=True)

    merged = pd.merge_asof(
        df_base_sorted,
        df_spo2,
        on="timestamp_naive",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=MATCH_TOLERANCE_MINUTES),
    )

    n_matched = merged["oxygen_saturation"].notna().sum()
    n_total = len(merged)
    match_pct = 100 * n_matched / n_total if n_total else 0.0
    print(f"[+] Matched {n_matched}/{n_total} grid timesteps to a real SpO2 reading "
          f"within {MATCH_TOLERANCE_MINUTES} min ({match_pct:.1f}%).")

    if match_pct < 30:
        print("[!] WARNING: fewer than 30% of timesteps matched. has_spo2 will still be")
        print("    True dataset-wide (column exists), but most individual windows will")
        print("    have interpolated/filled rather than freshly-measured values.")
        print("    Consider widening MATCH_TOLERANCE_MINUTES or checking whether the")
        print("    device actually samples SpO2 this sparsely for this patient.")

    merged = merged.drop(columns=["timestamp_naive"])
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"[SAVED] Merged dataset saved to: {OUTPUT_CSV}")
    print("\n[!] REMINDER: this is a new file. Confirm it looks right, then copy/rename")
    print("    it to whatever path window_generator.py's get_dataloader() actually reads")
    print("    (currently defaults to results/cohort/patient_1031_merged.csv).")


if __name__ == "__main__":
    merge_spo2_to_patient_csv()