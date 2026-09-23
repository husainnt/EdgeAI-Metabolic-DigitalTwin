"""
Confirms or denies the warm-start scale-blowup hypothesis for the three
standout patients (7751, 7188, 7392), by directly printing the scale factor
computed at EVERY calibration point:
    scale = target_bg / current_bg
from warm_start() in run_selective_warmstart_simglucose_hybrid.py.

A scale factor far from 1.0 (say, >3x or <0.3x) for the specific segment
that showed catastrophic error would confirm this is the root cause -- the
proportional rescale sent the simulator's compartments to an implausible
state for that one segment.

Usage:
    python check_warmstart_scale_blowup.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "glycemic_twin", "ml_layer")))
from window_generator import extract_2x_daily_calibrations
from run_selective_warmstart_simglucose_hybrid import build_calibrated_t2d_patient, get_Vg

COHORT_CSV_PATTERN = "results/cohort/patient_{pid}_merged.csv"

# The specific bad segment start index found for each patient (first index
# of the worst-error block from inspect_standout_patients.py)
BAD_SEGMENT_START = {"7751": 1958, "7188": 370, "7392": 688}


def check_patient(pid: str):
    csv_path = COHORT_CSV_PATTERN.format(pid=pid)
    if not os.path.exists(csv_path):
        print(f"[!] {csv_path} not found, skipping {pid}")
        return

    df = pd.read_csv(csv_path)
    glucose = df["glucose_mg_dl"].values.astype(float)
    calib_indices = extract_2x_daily_calibrations(df)

    patient = build_calibrated_t2d_patient()
    bad_idx = BAD_SEGMENT_START.get(pid)

    print(f"\n=== Patient {pid} ===")
    print(f"{'Calib idx':>10} | {'g_calib (real)':>15} | {'sim current_bg':>15} | {'scale factor':>13} | flag")

    for k, idx_start in enumerate(calib_indices):
        g_calib = glucose[idx_start]

        # Here I reset the patient exactly like the warm_start()/the main pipeline
        # does, then compute the same scale factor it would have computed
        patient.reset()
        Vg = get_Vg(patient)
        current_bg = patient.state[3] / Vg
        scale = g_calib / current_bg if current_bg != 0 else float("inf")

        is_bad_segment = idx_start <= bad_idx < (calib_indices[k + 1] if k + 1 < len(calib_indices) else len(glucose))
        flag = "  <-- BAD SEGMENT (matches inspect_standout_patients.py)" if is_bad_segment else ""
        extreme = " [EXTREME SCALE]" if (scale > 3.0 or scale < 0.3) else ""

        print(f"{idx_start:>10} | {g_calib:>15.1f} | {current_bg:>15.1f} | {scale:>13.3f}{extreme}{flag}")


if __name__ == "__main__":
    for pid in BAD_SEGMENT_START:
        check_patient(pid)