"""
Extracts real Garmin Activity telemetry (OMH 'activity' schema) and merges
it onto Patient 1031's existing 5-minute grid dataset.

Schema confirmed via Debug_activity_schema.py and inspect_activity_data_quality.py:
    body['activity'] = [
        {'activity_name': 'sedentary' | 'walking' | 'generic' | '',
         'base_movement_quantity': {'value': <steps, str-castable float>, 'unit': 'steps'},
         'effective_time_frame': {'time_interval': {'start_date_time', 'end_date_time'}}},
        ...
    ]
Data quality confirmed clean: 100.0% of entries (4,354/4,354 minus 2 empty)
carry a real numeric step value for Patient 1031.

UNLIKE SpO2 (point-in-time, nearest-match), Activity entries represent
short epochs that should ACCUMULATE within a 5-minute window, not be
nearest-matched to a single grid point. This bins raw entries into the
existing (irregularly-spaced but ~5-min) grid using each grid row's
preceding timestamp as the bin's lower edge, and sums steps + computes a
walking-fraction within each bin -- giving TWO genuinely real features
(no adapt_seq_dim tiling needed, since enc_activity expects exactly
in_features=2).

Produces two new columns:
    steps_sum               -- total steps summed across all raw entries
                                falling in this bin
    activity_walking_frac   -- fraction of raw entries in this bin whose
                                activity_name == 'walking' (0.0 if the bin
                                had zero raw entries at all)

Also reports, separately from the values above, what fraction of grid bins
had ZERO raw entries assigned at all (a genuine coverage gap) vs bins with
entries but zero steps (a real "no movement recorded" reading) -- these are
different things and conflating them would hide information the same way
the sleep/SpO2 body-key mismatches did earlier in this project.

Usage:
    python extract_and_merge_activity.py
"""

import os
import glob
import json
import numpy as np
import pandas as pd

INPUT_CSV = "results/cohort/patient_1031_merged.csv"
OUTPUT_CSV = "results/patient_1031_real_cgm_hr_steps_sleep_spo2_activity.csv"

ACTIVITY_DATA_DIR = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset\wearable_activity_monitor\physical_activity\garmin_vivosmart5\1031"


def load_activity_entries():
    pattern = os.path.join(ACTIVITY_DATA_DIR, "*.json")
    activity_files = glob.glob(pattern)
    print(f"[i] Searching {ACTIVITY_DATA_DIR}...")
    print(f"[i] Found {len(activity_files)} activity JSON files.")

    # Here I load every raw activity entry across all JSON files for this patient
    records = []
    for file_path in activity_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            entries = data.get("body", {}).get("activity", [])
            for entry in entries:
                bmq = entry.get("base_movement_quantity", {})
                raw_val = bmq.get("value", "")
                # Here I skip entries with no usable step value at all (confirmed rare: ~0.05%)
                if raw_val == "" or raw_val is None:
                    continue
                try:
                    steps_val = float(raw_val)
                except (TypeError, ValueError):
                    continue

                tf = entry.get("effective_time_frame", {}).get("time_interval", {})
                start_str = tf.get("start_date_time")
                if start_str is None:
                    continue

                # Here I strip timezone info so this lines up with the base CSV's naive timestamps
                ts = pd.to_datetime(start_str).tz_localize(None)
                act_name = entry.get("activity_name", "")
                records.append((ts, steps_val, act_name))

        except Exception as e:
            print(f"[!] Error reading {file_path}: {e}")
            continue

    print(f"[+] Extracted {len(records)} usable raw activity entries.")
    return records


def bin_activity_onto_grid(df_base, records):
    grid_ts = pd.to_datetime(df_base["timestamp"]).dt.tz_localize(None).values.astype("datetime64[ns]")
    n_grid = len(grid_ts)

    raw_ts = np.array([r[0] for r in records], dtype="datetime64[ns]")
    raw_steps = np.array([r[1] for r in records], dtype=np.float64)
    raw_is_walk = np.array([r[2] == "walking" for r in records], dtype=np.float64)

    # Here I bin each raw entry into the grid slot it falls into: (previous grid
    # timestamp, current grid timestamp]. searchsorted with side='left'
    # gives the index of the first grid_ts >= raw_ts, which is exactly that
    # right-closed bin under a monotonically increasing, ~uniformly-spaced grid.
    bin_idx = np.searchsorted(grid_ts, raw_ts, side="left")
    valid = (bin_idx >= 0) & (bin_idx < n_grid)

    # Here I accumulate steps and walking-entry counts per bin using np.add.at,
    # since multiple raw entries can land in the same 5-min bin
    steps_sum = np.zeros(n_grid, dtype=np.float64)
    walk_count = np.zeros(n_grid, dtype=np.float64)
    total_count = np.zeros(n_grid, dtype=np.float64)

    np.add.at(steps_sum, bin_idx[valid], raw_steps[valid])
    np.add.at(walk_count, bin_idx[valid], raw_is_walk[valid])
    np.add.at(total_count, bin_idx[valid], 1.0)

    # Here I compute the walking-fraction per bin, guarding against divide-by-zero
    # for bins that got zero raw entries at all
    walking_frac = np.divide(
        walk_count, total_count,
        out=np.zeros_like(walk_count),
        where=total_count > 0
    )

    n_zero_entry_bins = int((total_count == 0).sum())
    n_entry_but_zero_steps = int(((total_count > 0) & (steps_sum == 0)).sum())

    print(f"[i] Grid bins with ZERO raw entries assigned (genuine coverage gap): "
          f"{n_zero_entry_bins}/{n_grid} ({100*n_zero_entry_bins/n_grid:.1f}%)")
    print(f"[i] Grid bins WITH entries but zero total steps (real 'no movement' reading): "
          f"{n_entry_but_zero_steps}/{n_grid} ({100*n_entry_but_zero_steps/n_grid:.1f}%)")

    return steps_sum, walking_frac


def merge_activity_to_patient_csv():
    if not os.path.exists(INPUT_CSV):
        print(f"[!] Base CSV missing: {INPUT_CSV}")
        return

    df_base = pd.read_csv(INPUT_CSV)
    print(f"[i] Loaded {INPUT_CSV}: {len(df_base):,} rows, columns: {df_base.columns.tolist()}")

    records = load_activity_entries()
    if not records:
        raise RuntimeError(
            f"[!] CRITICAL FAILURE: Zero usable activity entries extracted from "
            f"{ACTIVITY_DATA_DIR}.\nCheck schema parsing logic before proceeding."
        )

    steps_sum, walking_frac = bin_activity_onto_grid(df_base, records)

    df_base["steps_sum"] = steps_sum
    df_base["activity_walking_frac"] = walking_frac

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_base.to_csv(OUTPUT_CSV, index=False)
    print(f"[SAVED] Merged dataset saved to: {OUTPUT_CSV}")
    print(f"\nSteps_sum summary: mean={steps_sum.mean():.1f}, max={steps_sum.max():.1f}")
    print(f"Walking-fraction summary: mean={walking_frac.mean():.3f}, "
          f"nonzero bins={int((walking_frac > 0).sum())}")
    print("\n[!] REMINDER: this is a new file, NOT yet the cohort training path.")
    print("    Confirm it looks right, then run prepare_cohort_file_with_activity.py")
    print("    to consolidate it into results/cohort/patient_1031_merged.csv.")


if __name__ == "__main__":
    merge_activity_to_patient_csv()