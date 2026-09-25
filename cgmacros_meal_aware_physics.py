"""
cgmacros_meal_aware_physics.py
==========================================================
Meal-aware simglucose physics baseline for CGMacros, built from the SAME
validated engine pieces as run_selective_warmstart_simglucose_hybrid.py
(build_calibrated_t2d_patient, warm_start, T2DPancreaticController) --
imported, not reimplemented, so the T2D parameter lock (Vmx x0.75,
kp3 x0.70) and the insulin-damping warm-start fix are identical to every
other physics number in this project.

This module does NOT modify run_selective_warmstart_simglucose_hybrid.py,
so the Patient 1031 AI-READI regression numbers (physics RMSE 49.54,
hybrid RMSE 33.35-34.45 across seeds) are unaffected by anything here.

Design: CGMacros gives a native 1-minute grid, and simglucose's own
sample_time for adult#001 is 1 minute -- so unlike the AI-READI pipeline
(which steps 5 sub-steps per 5-min output row), this steps the patient
ONE substep per CGMacros row, giving meal timing exact-minute precision
rather than snapping to a 5-minute bin.

Two baselines are computed with the EXACT same code path, differing only
in whether real logged carbs are injected as CHO at the meal-onset
minute (use_meals=True) or CHO is held at 0 throughout (use_meals=False,
matching every other physics baseline in this project). This is what
makes the postprandial-window comparison apples-to-apples: any
difference between the two comes only from the meal signal, not from a
different stepping scheme.

Calibration points are extracted the same way window_generator.py does
for AI-READI (~8am/8pm real readings, 45-min tolerance) but reading
CGMacros' real Dexcom/Libre glucose column as the "fingerstick" source --
CGMacros technically has continuous CGM, but the project's whole premise
is CGM-free deployment, so only sparse anchor points are used here,
consistent with the AI-READI mechanistic baseline.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from run_selective_warmstart_simglucose_hybrid import (
    build_calibrated_t2d_patient, warm_start, T2DPancreaticController
)
from simglucose.patient.t1dpatient import Action as PatientAction


def extract_2x_daily_calibrations_cgmacros(df, glucose_col="Dexcom GL", timestamp_col="Timestamp",
                                            target_hours=(8, 20), window_tolerance_min=45):
    """Same ~8am/8pm sparse-anchor pattern as the AI-READI pipeline, reading
    CGMacros' own CGM column as the stand-in fingerstick source. Falls back
    row-by-row to Libre GL where Dexcom GL is NaN (CGMacros ships both)."""
    ts = pd.to_datetime(df[timestamp_col])
    glucose = df[glucose_col].copy()
    if "Libre GL" in df.columns:
        glucose = glucose.fillna(df["Libre GL"])
    dates = ts.dt.date.unique()
    indices = set()
    for d in dates:
        day_mask = ts.dt.date == d
        for hour in target_hours:
            target = pd.Timestamp(f"{d} {hour:02d}:00:00")
            diffs = (ts[day_mask] - target).abs()
            valid = diffs[day_mask & glucose.notna()] if False else diffs[glucose[day_mask].notna()]
            if len(valid) == 0:
                continue
            best_idx = valid.idxmin()
            if valid.loc[best_idx] <= pd.Timedelta(minutes=window_tolerance_min):
                indices.add(best_idx)
    return sorted(indices), glucose.values.astype(np.float32)


def build_meal_onset_cho(df):
    """Here I place each real meal's scaled carbs ONLY at its onset row
    (time_since_last_meal_min == 0), not across the forward-filled rows --
    the forward fill in extract_cgmacros_diet_features.py is a CONTEXT
    feature (what was last eaten), not a repeated-ingestion signal. Feeding
    it every row would make the model think the patient eats continuously."""
    cho = np.zeros(len(df), dtype=np.float32)
    onset = df["time_since_last_meal_min"].values == 0
    cho[onset] = df.loc[onset, "Carbs_eaten"].values.astype(np.float32)
    return cho


def _run_segment(patient, controller, n_rows, cho_by_row):
    """Steps ONE substep per CGMacros row (native 1-min grid = simglucose's
    own sample_time for adult#001). Returns the per-row plasma glucose
    trajectory. Identical code path regardless of whether cho_by_row is
    all-zero (meal-blind) or has real meals in it (meal-aware)."""
    Vg = patient.state[3] / patient._params["Gb"]
    traj = np.zeros(n_rows, dtype=np.float32)
    for i in range(n_rows):
        bg_now = patient.state[3] / Vg
        basal = controller.basal_for(bg_now)
        patient.step(PatientAction(CHO=float(cho_by_row[i]), insulin=basal))
        traj[i] = patient.state[3] / Vg
    return traj


def compute_meal_aware_baseline(df, glucose_col="Dexcom GL", use_meals=True):
    """Returns (sim_baseline, calib_indices). sim_baseline is per-row
    (1-min resolution). Pre-first-calibration rows are filled with the
    first real glucose value, same convention as window_generator.py."""
    calib_indices, glucose = extract_2x_daily_calibrations_cgmacros(df, glucose_col)
    if len(calib_indices) < 2:
        raise ValueError("Not enough calibration points to build a mechanistic baseline for this subject/segment.")

    cho_by_row = build_meal_onset_cho(df) if use_meals else np.zeros(len(df), dtype=np.float32)

    patient = build_calibrated_t2d_patient()
    params = patient._params
    basal_rate = params["u2ss"] * params["BW"] / 6000
    controller = T2DPancreaticController(gb=params["Gb"], basal_rate=basal_rate)

    sim_baseline = np.zeros(len(df), dtype=np.float32)
    for k in range(len(calib_indices)):
        idx_start = calib_indices[k]
        idx_end = calib_indices[k + 1] if k + 1 < len(calib_indices) else len(df)
        n_rows = idx_end - idx_start
        g_calib = glucose[idx_start]
        if not np.isfinite(g_calib):
            continue  # Here I skip a segment whose anchor reading is missing from BOTH Dexcom and Libre

        patient.reset()
        # Here I re-apply the T2D lock defensively, same guard pattern as
        # the AI-READI script, in case a future simglucose version resets it
        patient._params["Vmx"] = params["Vmx"]
        patient._params["kp3"] = params["kp3"]
        warm_start(patient, g_calib)
        sim_baseline[idx_start:idx_end] = _run_segment(patient, controller, n_rows, cho_by_row[idx_start:idx_end])

    if calib_indices[0] > 0:
        sim_baseline[: calib_indices[0]] = glucose[0]
    return sim_baseline, calib_indices


def postprandial_gate_report(df, glucose_col="Dexcom GL", window_hours=3):
    """The gate: meal-aware physics must beat meal-blind physics in the
    window_hours after each meal onset, or meal-awareness isn't earning
    its complexity. Runs BOTH baselines through the identical stepping
    code, differing only in cho_by_row, so this is a clean A/B test."""
    real = df[glucose_col].fillna(df.get("Libre GL")).values.astype(np.float32)
    aware, calib_idx = compute_meal_aware_baseline(df, glucose_col, use_meals=True)
    blind, _ = compute_meal_aware_baseline(df, glucose_col, use_meals=False)

    onset_rows = df.index[df["time_since_last_meal_min"].values == 0].tolist()
    win = int(window_hours * 60)  # 1-min grid
    aware_sq, blind_sq, n = 0.0, 0.0, 0
    for r in onset_rows:
        lo, hi = r, min(r + win, len(df))
        seg = real[lo:hi]
        valid = np.isfinite(seg) & np.isfinite(aware[lo:hi]) & np.isfinite(blind[lo:hi])
        if valid.sum() == 0:
            continue
        aware_sq += np.sum((aware[lo:hi][valid] - seg[valid]) ** 2)
        blind_sq += np.sum((blind[lo:hi][valid] - seg[valid]) ** 2)
        n += valid.sum()
    if n == 0:
        return None
    rmse_aware = np.sqrt(aware_sq / n)
    rmse_blind = np.sqrt(blind_sq / n)
    verdict = "PASSES gate (meal-aware beats meal-blind)" if rmse_aware < rmse_blind else "FAILS gate"
    print(f"Postprandial ({window_hours}h) RMSE -- meal-blind: {rmse_blind:.2f} | "
          f"meal-aware: {rmse_aware:.2f} | n_meals: {len(onset_rows)} | n_points: {n} | {verdict}")
    return {"rmse_blind": rmse_blind, "rmse_aware": rmse_aware, "n_meals": len(onset_rows),
            "n_points": int(n), "passes": bool(rmse_aware < rmse_blind)}