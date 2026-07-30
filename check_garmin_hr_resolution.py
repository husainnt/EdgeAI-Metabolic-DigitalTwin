"""
Inspect raw Garmin heart rate sample gaps for Patient 1031 to determine
if true beat-to-beat (R-R) data exists or if it's periodic averaged HR.
"""

import json
import os
import pandas as pd

RAW_GARMIN_PATH = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset\wearable_activity_monitor\heart_rate\garmin_vivosmart5\1031\1031_heartrate.json"


def inspect_garmin_resolution():
    if not os.path.exists(RAW_GARMIN_PATH):
        print(f"[!] Path not found: {RAW_GARMIN_PATH}")
        return

    with open(RAW_GARMIN_PATH, "r") as f:
        raw = json.load(f)

    # Access OMH schema nested array: body -> heart_rate
    items = raw.get("body", {}).get("heart_rate", [])
    print(f"[+] Total heart rate entries found: {len(items):,}")

    if not items:
        print("[!] No items found inside raw['body']['heart_rate'].")
        return

    timestamps = []
    for entry in items:
        if not isinstance(entry, dict):
            continue

        ts = None
        eff = entry.get("effective_time_frame", {})
        if isinstance(eff, dict):
            ts = eff.get("date_time") or eff.get("date_time_start") or eff.get("date_time_end")

        if not ts and "date_time" in entry:
            ts = entry["date_time"]

        if ts:
            timestamps.append(pd.to_datetime(ts))

    timestamps = sorted(list(set(timestamps)))
    print(f"[+] Total unique timestamps parsed: {len(timestamps):,}")

    if len(timestamps) < 2:
        print("[!] Not enough timestamps parsed. First sample entry:")
        print(items[0])
        return

    diffs = pd.Series(timestamps).diff().dropna()

    print("\n" + "=" * 55)
    print("      GARMIN HR SAMPLING RESOLUTION DIAGNOSTIC      ")
    print("=" * 55)
    print(f"Median gap between consecutive HR samples: {diffs.median()}")
    print(f"Min gap: {diffs.min()}")
    print(f"Max gap: {diffs.max()}")
    print("\nMost common sample time gaps:")
    print(diffs.value_counts().head(10))
    print("=" * 55)


if __name__ == "__main__":
    inspect_garmin_resolution()