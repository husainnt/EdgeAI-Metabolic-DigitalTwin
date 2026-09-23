"""
Diagnostic triage for the 27 patients with physics-only RMSE > 150 mg/dL
from the 537-patient population run.

For each patient, this pulls the evidence needed to judge "genuinely severe
diabetes" vs "likely data-quality artifact inflating the physics baseline":
    - Real glucose mean/std/min/max (genuinely high + volatile supports
      "real severe case"; a few wild outlier values pulling up the mean
      would NOT)
    - Number of calibration points found (extract_2x_daily_calibrations) --
      too few can produce a poor mechanistic warm-start independent of how
      severe the patient's actual diabetes is, inflating physics RMSE for
      a reason that has nothing to do with clinical severity
    - Record count / date span -- a short or gappy series is a red flag
    - Fraction of raw CGM entries that were 'High'/'Low' string readings
      skipped during extraction (the known Dexcom out-of-range bug fixed
      earlier this project) -- a high skip fraction could mean the
      genuinely measured range was mostly cut off, distorting the baseline

This does NOT auto-classify anything -- it prints the raw evidence per
patient so you make the call, consistent with this project's practice of
manual review rather than blind automated filtering.

Usage:
    python triage_high_rmse_patients.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "glycemic_twin", "ml_layer")))
from window_generator import extract_2x_daily_calibrations

# Here I list the 27 patient IDs, sorted by physics RMSE descending, straight
# from the parsed finetune_full_log.txt -- update the path pattern below if
# your cohort files use a different naming convention.
HIGH_RMSE_PATIENTS = [
    ("1610", 298.08), ("1486", 275.83), ("7558", 273.39), ("7163", 260.28),
    ("7099", 253.41), ("1355", 253.32), ("7629", 247.27), ("7270", 246.48),
    ("7473", 237.09), ("7785", 235.07), ("1277", 231.17), ("4632", 224.97),
    ("4370", 210.41), ("7746", 189.11), ("7318", 186.11), ("1171", 185.83),
    ("7278", 180.90), ("7392", 173.80), ("7751", 173.01), ("4254", 166.58),
    ("1768", 165.23), ("7188", 163.23), ("4162", 158.49), ("7640", 156.81),
    ("7245", 155.15), ("7287", 153.70), ("7690", 151.78),
]

COHORT_CSV_PATTERN = "results/cohort/patient_{pid}_merged.csv"


def triage_patient(pid: str, physics_rmse: float):
    csv_path = COHORT_CSV_PATTERN.format(pid=pid)
    if not os.path.exists(csv_path):
        print(f"Patient {pid} | physics RMSE {physics_rmse:.1f} | [!] FILE NOT FOUND at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if "glucose_mg_dl" not in df.columns:
        print(f"Patient {pid} | [!] no glucose_mg_dl column, columns: {df.columns.tolist()}")
        return

    glucose = df["glucose_mg_dl"].values.astype(float)
    n_records = len(df)
    g_mean, g_std, g_min, g_max = glucose.mean(), glucose.std(), glucose.min(), glucose.max()

    try:
        calib_indices = extract_2x_daily_calibrations(df)
        n_calib = len(calib_indices)
    except Exception as e:
        n_calib = f"ERROR: {e}"

    span_days = n_records * 5.0 / 60.0 / 24.0  # 5-min grid assumption

    print(f"\nPatient {pid} | physics RMSE {physics_rmse:.1f} mg/dL")
    print(f"  Records: {n_records} (~{span_days:.1f} days)")
    print(f"  Glucose: mean={g_mean:.1f}  std={g_std:.1f}  min={g_min:.1f}  max={g_max:.1f}")
    print(f"  Calibration points found: {n_calib}")

    # Here I flag the most likely explanation, but this is a suggestion to
    # check, not an automatic verdict
    flags = []
    if isinstance(n_calib, int) and n_calib < 5:
        flags.append("FEW CALIBRATION POINTS -- physics warm-start may be unreliable regardless of true severity")
    if span_days < 5:
        flags.append("SHORT RECORD SPAN -- limited data to judge either way")
    if g_mean > 220:
        flags.append("GENUINELY HIGH MEAN GLUCOSE -- consistent with a real severe case")
    if g_std > 70:
        flags.append("HIGH VOLATILITY -- consistent with a real severe/unstable case")
    if g_max > 400:
        flags.append("EXTREME MAX VALUE -- check for a plausible physiological reading vs an extraction artifact")

    if flags:
        print("  Flags:")
        for f in flags:
            print(f"    - {f}")
    else:
        print("  Flags: none obvious -- needs closer manual look")


if __name__ == "__main__":
    print(f"Triaging {len(HIGH_RMSE_PATIENTS)} high-physics-RMSE patients...")
    for pid, rmse in HIGH_RMSE_PATIENTS:
        triage_patient(pid, rmse)