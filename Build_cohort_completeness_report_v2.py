"""
FYP-II Cohort Completeness Report
==================================

Purpose:
    Builds the real oral-medication T2D cohort (verified from clinical_data,
    not assumed) and reports, per patient, whether each wearable modality
    (HR, CGM, Sleep, SpO2) is present and how many records it has.

    This is NOT a hard all-or-nothing filter. It reports full modality
    availability so you can later choose whichever modality combination
    a given experiment needs (e.g. HR+CGM only, HR+CGM+Sleep, full stack),
    matching the ablation-style approach already used in the experiment log
    (context-only, HR-alone, sleep+context, etc.).

Cohort derivation (verified against real files, 2026-09):
    - T2D diagnosis: condition_occurrence.csv, condition_source_value
      starting with "mhterm_dm2" (self-reported Type II Diabetes)
    - Oral-medication-only: observation.csv, observation_source_value
      starting with "cmtrt_insln" (insulin question), value_as_number == 0
    - Verified counts: 899 T2D diagnosed, 646 oral-med-only (not on insulin),
      253 on insulin (excluded). NOTE: this is 646, not the earlier
      remembered 686 -- 646 is the number backed by actual file contents,
      use this going forward and mention the correction to supervisors.

Usage:
    Update DATASET_BASE and CLINICAL_DATA_DIR below to match your local
    paths, then run:
        python build_cohort_completeness_report.py

Output:
    cohort_completeness_report.csv in the same directory as this script,
    with one row per oral-med T2D patient and modality availability columns.
"""

import os
import csv
import json

# ---------------------------------------------------------------------------
# CONFIG -- update these paths to match your machine
# ---------------------------------------------------------------------------
DATASET_BASE = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset"
CLINICAL_DATA_DIR = os.path.join(DATASET_BASE, "clinical_data")

CONDITION_OCCURRENCE_CSV = os.path.join(CLINICAL_DATA_DIR, "condition_occurrence.csv")
OBSERVATION_CSV = os.path.join(CLINICAL_DATA_DIR, "observation.csv")

# Wearable folders (confirmed nested OMH-schema JSON structure, NOT flat CSVs)
HR_DIR = os.path.join(DATASET_BASE, "wearable_activity_monitor", "heart_rate", "garmin_vivosmart5")
SLEEP_DIR = os.path.join(DATASET_BASE, "wearable_activity_monitor", "sleep", "garmin_vivosmart5")
SPO2_DIR = os.path.join(DATASET_BASE, "wearable_activity_monitor", "oxygen_saturation", "garmin_vivosmart5")
ACTIVITY_DIR = os.path.join(DATASET_BASE, "wearable_activity_monitor", "physical_activity", "garmin_vivosmart5")
CGM_DIR = os.path.join(DATASET_BASE, "wearable_blood_glucose", "continuous_glucose_monitoring", "dexcom_g6")

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cohort_completeness_report.csv")


# ---------------------------------------------------------------------------
# STEP 1: Derive the real oral-medication T2D cohort from clinical_data
# ---------------------------------------------------------------------------
def get_t2d_oral_med_cohort():
    """Returns (t2d_ids, oral_med_ids, insulin_ids) as sets of person_id strings."""
    t2d_ids = set()
    with open(CONDITION_OCCURRENCE_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_val = row.get("condition_source_value", "")
            if source_val.startswith("mhterm_dm2"):
                t2d_ids.add(row["person_id"])

    insulin_status = {}
    with open(OBSERVATION_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_val = row.get("observation_source_value", "")
            if source_val.startswith("cmtrt_insln"):
                insulin_status[row["person_id"]] = row.get("value_as_number", "")

    oral_med_ids = {pid for pid in t2d_ids if insulin_status.get(pid) == "0"}
    insulin_ids = {pid for pid in t2d_ids if insulin_status.get(pid) == "1"}

    print(f"[+] T2D diagnosed (mhterm_dm2): {len(t2d_ids)}")
    print(f"[+] Oral-medication-only (not on insulin): {len(oral_med_ids)}")
    print(f"[+] On insulin (excluded from cohort): {len(insulin_ids)}")

    return t2d_ids, oral_med_ids, insulin_ids


# ---------------------------------------------------------------------------
# STEP 2: Per-modality presence + count checks
# ---------------------------------------------------------------------------
def count_json_entries(json_path, body_key, list_key_candidates):
    """
    Opens an OMH-schema JSON file and counts entries in body[body_key].
    list_key_candidates: list of possible keys to try under body[body_key]
    (schemas vary slightly across modalities -- try each until one works).
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        body = data.get("body", {})
        entries = body.get(body_key, [])
        if entries:
            return len(entries)
        # try alternate keys if body_key itself was wrong/empty
        for k in list_key_candidates:
            entries = body.get(k, [])
            if entries:
                return len(entries)
        return 0
    except Exception:
        return 0


def check_modality(base_dir, pid, body_key, alt_keys=None):
    """
    Checks whether patient `pid` has a folder + JSON file under `base_dir`,
    and returns (present: bool, count: int).
    """
    alt_keys = alt_keys or []
    pid_dir = os.path.join(base_dir, pid)
    if not os.path.isdir(pid_dir):
        return False, 0

    json_files = [f for f in os.listdir(pid_dir) if f.endswith(".json")]
    if not json_files:
        return False, 0

    total_count = 0
    for jf in json_files:
        total_count += count_json_entries(os.path.join(pid_dir, jf), body_key, alt_keys)

    return total_count > 0, total_count


def find_cgm_count(pid):
    """CGM lives under continuous_glucose_monitoring/dexcom_g6/<pid>/*.json"""
    pid_dir = os.path.join(CGM_DIR, pid)
    if not os.path.isdir(pid_dir):
        return False, 0
    json_files = [f for f in os.listdir(pid_dir) if f.endswith(".json")]
    if not json_files:
        return False, 0

    total = 0
    for jf in json_files:
        path = os.path.join(pid_dir, jf)
        try:
            with open(path, "r") as f:
                raw = json.load(f)
            entries = []
            if isinstance(raw, dict):
                entries = raw.get("body", {}).get("cgm", []) or raw.get("body", {}).get("blood_glucose", [])
            elif isinstance(raw, list):
                entries = raw
            total += len(entries)
        except Exception:
            continue
    return total > 0, total


# ---------------------------------------------------------------------------
# STEP 3: Build the report
# ---------------------------------------------------------------------------
def build_report():
    t2d_ids, oral_med_ids, insulin_ids = get_t2d_oral_med_cohort()

    rows = []
    for i, pid in enumerate(sorted(oral_med_ids)):
        hr_present, hr_count = check_modality(HR_DIR, pid, "heart_rate")
        sleep_present, sleep_count = check_modality(SLEEP_DIR, pid, "sleep")
        spo2_present, spo2_count = check_modality(SPO2_DIR, pid, "breathing")
        activity_present, activity_count = check_modality(ACTIVITY_DIR, pid, "activity")
        cgm_present, cgm_count = find_cgm_count(pid)

        rows.append({
            "person_id": pid,
            "hr_present": hr_present, "hr_count": hr_count,
            "cgm_present": cgm_present, "cgm_count": cgm_count,
            "sleep_present": sleep_present, "sleep_count": sleep_count,
            "spo2_present": spo2_present, "spo2_count": spo2_count,
            "activity_present": activity_present, "activity_count": activity_count,
        })

        if (i + 1) % 50 == 0:
            print(f"    ...processed {i + 1}/{len(oral_med_ids)} patients")

    with open(OUTPUT_CSV, "w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[+] Report written to {OUTPUT_CSV}")

    # Summary counts for common modality-combination requirements
    def count_where(pred):
        return sum(1 for r in rows if pred(r))

    print("\n" + "=" * 60)
    print("COHORT SIZE UNDER DIFFERENT MODALITY REQUIREMENTS")
    print("=" * 60)
    print(f"Oral-med T2D total:                        {len(rows)}")
    print(f"+ HR present:                               {count_where(lambda r: r['hr_present'])}")
    print(f"+ HR + CGM:                                 {count_where(lambda r: r['hr_present'] and r['cgm_present'])}")
    print(f"+ HR + CGM + Sleep:                         {count_where(lambda r: r['hr_present'] and r['cgm_present'] and r['sleep_present'])}")
    print(f"+ HR + CGM + Sleep + SpO2:                  {count_where(lambda r: r['hr_present'] and r['cgm_present'] and r['sleep_present'] and r['spo2_present'])}")
    print(f"+ HR + CGM + Sleep + SpO2 + Activity (all): {count_where(lambda r: all([r['hr_present'], r['cgm_present'], r['sleep_present'], r['spo2_present'], r['activity_present']]))}")
    print("=" * 60)


if __name__ == "__main__":
    build_report()