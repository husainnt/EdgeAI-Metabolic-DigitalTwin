"""
build_cgmacros_window_cache.py
==========================================================
Builds a per-subject cache of six-encoder training windows for the CGMacros
T2D cohort (diet track), aligned EXACTLY like GlycemicWindowDataset in
window_generator.py (the AI-READI pipeline):

    12 input steps on a 5-min grid, horizon 6 steps (30 min)
    curr_idx   = i + 12 - 1          -> mech_curr read here
    target_idx = i + 12 + 6 - 1      -> y, mech_fore, and the calibration
                                        context (tsc hours, last anchor mg/dL)
                                        are all read here
    a window is skipped if y or any of the 12 input CGM values is NaN, or if
    the physics baseline is exactly 0 at curr/target (outside any warm-started
    segment) -- same rules as the generator

CGMacros differences (all deliberate, all disclosed in the cache):
    * native 1-min rows are reduced to the 5-min grid by TRAILING bins (each
      grid step uses the 5 rows ending at it, so nothing looks ahead)
    * physics runs at native 1-min over the whole record, ONCE per variant,
      via cgmacros_meal_aware_physics.compute_meal_aware_baseline; windows
      just index into those trajectories
    * the TRAINING baseline is the MEAL-BLIND physics (mech_curr/mech_fore);
      the meal-aware trajectories are stored separately, diagnostics only
    * Activity channels are [METs (5-min mean), Calories (Activity) (5-min
      sum)] instead of [steps, walking fraction]; Sleep and SpO2 do not exist
      in CGMacros, so they are zeros with has=0
    * Diet vector (7, matching enc_diet in_features) is read at curr_idx, the
      last input step. Reading it at target_idx would leak a meal eaten
      during the forecast horizon.

Where it lives: D:\\FYP\\CODE. Run it from there.

Inputs (from extract_cgmacros_diet_features.py):
    results/cgmacros_cohort/cgmacros_<pid:03d>_diet_features.csv
Outputs:
    results/cgmacros_window_cache/subject_<pid>.npz
    results/cgmacros_window_cache/build_status.csv

Keys in each .npz (N = windows kept, time-ordered; values RAW, unscaled):
    hr (N,12)  act (N,12,2)  sleep (N,12)  spo2 (N,12)  glc (N,2)
    diet (N,7)  mech_curr (N,)  mech_fore (N,)  y (N,)
    mech_curr_aware (N,)  mech_fore_aware (N,)      [diagnostics only]
    pre_calib (N,) bool   anchor_in_horizon (N,) bool   macro_capped (N,)
    tsm_target_min (N,)   curr_row (N,)  target_row (N,)   [row = 1-min row]
    has (4,) = [hr, act, sleep, spo2]   diet_cols   act_cols   patient_id

Resumable: a subject whose .npz exists is skipped (--force rebuilds). Files are
written to a temp name and renamed, so an interrupted run never leaves a
half-written cache file.
"""

import os
import re
import csv
import sys
import glob
import time
import argparse
import traceback
import warnings
from multiprocessing import Pool, freeze_support

import numpy as np
import pandas as pd

# Here I put this script's own folder on sys.path so the physics module (and
# through it run_selective_warmstart_simglucose_hybrid) imports the same way
# it does in check_cgmacros_gate.py
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

COHORT_DIR = os.path.join("results", "cgmacros_cohort")
OUT_DIR = os.path.join("results", "cgmacros_window_cache")
STATUS_CSV = os.path.join(OUT_DIR, "build_status.csv")

# Here I keep this in sync with extract_cgmacros_diet_features.py, since a
# stale feature file for an excluded subject can still be sitting in COHORT_DIR
EXCLUDE_SUBJECTS = {30}

FILE_RE = re.compile(r"^cgmacros_(\d+)_diet_features\.csv$")

WINDOW = 12   # input steps on the 5-min grid (60 min)
HORIZON = 6   # 6 x 5 min = 30 min ahead
BIN = 5       # 1-min rows per 5-min grid step
POSTPRANDIAL_MIN = 180  # same 3 h window the gate uses

# Here I fix the diet vector order to what enc_diet was widened for in
# hybrid_twin.py: 5 scaled macros + time_since_last_meal_min + has_had_meal
DIET_COLS = ["Carbs_eaten", "Protein_eaten", "Fat_eaten", "Fiber_eaten",
             "Calories_eaten", "time_since_last_meal_min", "has_had_meal"]
ACT_COLS = ["METs_mean5", "CaloriesActivity_sum5"]

STATUS_FIELDS = [
    "subject", "status", "rows", "frac_irregular_dt", "max_gap_min",
    "n_meals", "n_calib", "n_windows", "n_pre_calib", "n_anchor_in_horizon",
    "n_clean", "n_postprandial_clean", "n_dropped_nonfinite",
    "phys_blind_rmse_clean", "phys_aware_rmse_clean",
    "phys_blind_rmse_pp", "phys_aware_rmse_pp",
    "has_hr", "has_act", "disabled_modalities", "seconds", "error",
]


def _rmse(a, b):
    if len(a) == 0:
        return float("nan")
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sqrt(np.mean(d ** 2)))


def _bin_mean(x, n_bins):
    # Here I take the mean of the finite values in each trailing 5-row bin;
    # a bin with no finite value stays NaN instead of becoming a fake 0
    blk = x[: n_bins * BIN].reshape(n_bins, BIN).astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(blk, axis=1).astype(np.float32)


def _bin_sum(x, n_bins):
    blk = x[: n_bins * BIN].reshape(n_bins, BIN).astype(np.float64)
    cnt = np.isfinite(blk).sum(axis=1)
    s = np.nansum(blk, axis=1)
    s[cnt == 0] = np.nan
    return s.astype(np.float32)


def build_windows(df, glucose, sim_blind, sim_aware, calib_indices):
    """
    Pure numpy: turns 1-min rows + 1-min physics trajectories into windows
    aligned like GlycemicWindowDataset. Kept separate from the simglucose
    calls so the alignment can be tested without simglucose installed.
    Returns (arrays dict, info dict).
    """
    n = len(df)
    n_bins = n // BIN
    if n_bins < WINDOW + HORIZON:
        raise ValueError(f"record too short for one window ({n} rows)")
    g_rows = np.arange(n_bins) * BIN + (BIN - 1)  # last 1-min row of each grid step

    calib = np.asarray(calib_indices, dtype=np.int64)
    rows = np.arange(n)

    # Here I rebuild the calibration context on the 1-min grid exactly like
    # compute_simglucose_baseline does on the 5-min grid: hours since the most
    # recent anchor and that anchor's value; before the first anchor it counts
    # hours from the record start against glucose[0]
    last = np.searchsorted(calib, rows, side="right") - 1
    g_first = glucose[np.isfinite(glucose)][0] if np.isfinite(glucose).any() else np.nan
    tsc_1m = np.where(last >= 0, (rows - calib[last]) / 60.0, rows / 60.0).astype(np.float32)
    calib_1m = np.where(last >= 0, glucose[calib[last]], g_first).astype(np.float32)

    # Here I put every per-row signal onto the 5-min grid
    real_g = glucose[g_rows]
    hr_g = _bin_mean(df["HR"].values.astype(np.float32), n_bins) if "HR" in df.columns \
        else np.full(n_bins, np.nan, np.float32)
    mets_g = _bin_mean(df["METs"].values.astype(np.float32), n_bins) if "METs" in df.columns \
        else np.full(n_bins, np.nan, np.float32)
    cal_g = _bin_sum(df["Calories (Activity)"].values.astype(np.float32), n_bins) \
        if "Calories (Activity)" in df.columns else np.full(n_bins, np.nan, np.float32)
    blind_g, aware_g = sim_blind[g_rows], sim_aware[g_rows]
    tsc_g, calib_g = tsc_1m[g_rows], calib_1m[g_rows]
    diet_g = df[DIET_COLS].values.astype(np.float32)[g_rows]
    capped_g = df["macro_capped"].values.astype(np.float32)[g_rows] if "macro_capped" in df.columns \
        else np.zeros(n_bins, np.float32)
    tsm_g = df["time_since_last_meal_min"].values.astype(np.float32)[g_rows]

    J = n_bins - WINDOW - HORIZON + 1
    i = np.arange(J)
    curr_g = i + WINDOW - 1
    targ_g = i + WINDOW + HORIZON - 1

    def seq(arr):
        return np.stack([arr[i + k] for k in range(WINDOW)], axis=1)

    # Here I apply the generator's skip rules: NaN target, NaN in the 12-step
    # CGM window, or a physics baseline of exactly 0 at curr/target
    x_cgm_ok = np.isfinite(seq(real_g)).all(axis=1)
    y = real_g[targ_g]
    valid = x_cgm_ok & np.isfinite(y) & (blind_g[curr_g] != 0) & (blind_g[targ_g] != 0)
    sel = np.where(valid)[0]
    curr_g, targ_g = curr_g[sel], targ_g[sel]
    curr_row, targ_row = g_rows[curr_g], g_rows[targ_g]

    # Here I flag the two leakage-relevant cases exactly, from the anchor
    # indices themselves rather than the equality heuristic the AI-READI cache uses
    first_calib = calib[0]
    pre_calib = (curr_row < first_calib) | (targ_row < first_calib)
    n_in_horizon = (np.searchsorted(calib, targ_row, side="right")
                    - np.searchsorted(calib, curr_row, side="right"))
    anchor_in_horizon = n_in_horizon > 0

    arrays = {
        "hr": seq(hr_g)[sel],
        "act": np.stack([seq(mets_g)[sel], seq(cal_g)[sel]], axis=-1),
        "sleep": np.zeros((len(sel), WINDOW), np.float32),
        "spo2": np.zeros((len(sel), WINDOW), np.float32),
        "glc": np.stack([tsc_g[targ_g], calib_g[targ_g]], axis=-1),
        "diet": diet_g[curr_g],
        "mech_curr": blind_g[curr_g],
        "mech_fore": blind_g[targ_g],
        "y": y[sel],
        "mech_curr_aware": aware_g[curr_g],
        "mech_fore_aware": aware_g[targ_g],
        "pre_calib": pre_calib,
        "anchor_in_horizon": anchor_in_horizon,
        "macro_capped": capped_g[curr_g],
        "tsm_target_min": tsm_g[targ_g],
        "curr_row": curr_row.astype(np.int32),
        "target_row": targ_row.astype(np.int32),
    }
    has = {"hr": 1.0, "act": 1.0, "sleep": 0.0, "spo2": 0.0}
    return arrays, has


def sanitize(arrays, has):
    """Same policy as build_six_encoder_window_cache.py: a modality that is
    non-finite everywhere is switched OFF (zeros, has=0); remaining windows
    with any non-finite model input are DROPPED and counted, never zero-filled."""
    disabled = []
    for k in ("hr", "act"):
        if has[k] and not np.isfinite(arrays[k]).any():
            arrays[k] = np.zeros_like(arrays[k])
            has[k] = 0.0
            disabled.append(k)
    n = len(arrays["y"])
    keep = np.ones(n, dtype=bool)
    for k in ("hr", "act", "glc", "diet"):
        keep &= np.isfinite(arrays[k].reshape(n, -1)).all(axis=1)
    for k in ("mech_curr", "mech_fore", "y"):
        keep &= np.isfinite(arrays[k])
    n_dropped = int((~keep).sum())
    if n_dropped:
        arrays = {k: v[keep] for k, v in arrays.items()}
    return arrays, has, n_dropped, disabled


def process_subject(task):
    pid, csv_path, out_path = task
    t0 = time.time()
    status = {k: "" for k in STATUS_FIELDS}
    status["subject"] = pid
    try:
        # Here I import inside the worker so each spawned process loads the
        # simglucose engine itself (required for Windows multiprocessing)
        from cgmacros_meal_aware_physics import compute_meal_aware_baseline

        df = pd.read_csv(csv_path).reset_index(drop=True)
        status["rows"] = len(df)

        # Here I check the 1-min assumption physics relies on (one substep per
        # row); irregular timestamps would silently stretch or squeeze sim time
        dt_min = pd.to_datetime(df["Timestamp"]).diff().dt.total_seconds().values[1:] / 60.0
        status["frac_irregular_dt"] = round(float(np.mean(np.abs(dt_min - 1.0) > 1e-6)), 5)
        status["max_gap_min"] = round(float(np.nanmax(dt_min)), 2) if len(dt_min) else 0.0
        status["n_meals"] = int((df["time_since_last_meal_min"].values == 0).sum())

        # Here I run each physics variant ONCE over the whole 1-min record
        sim_blind, calib_idx = compute_meal_aware_baseline(df, use_meals=False)
        sim_aware, _ = compute_meal_aware_baseline(df, use_meals=True)
        status["n_calib"] = len(calib_idx)

        glucose = df["Dexcom GL"].copy() if "Dexcom GL" in df.columns else pd.Series(np.nan, index=df.index)
        if "Libre GL" in df.columns:
            glucose = glucose.fillna(df["Libre GL"])
        glucose = glucose.values.astype(np.float32)

        arrays, has = build_windows(df, glucose, sim_blind, sim_aware, calib_idx)
        status["n_windows"] = len(arrays["y"])
        arrays, has, n_dropped, disabled = sanitize(arrays, has)
        status["n_dropped_nonfinite"] = n_dropped
        status["disabled_modalities"] = ";".join(disabled)
        status["has_hr"], status["has_act"] = int(has["hr"]), int(has["act"])

        clean = ~(arrays["pre_calib"] | arrays["anchor_in_horizon"])
        pp = clean & (arrays["tsm_target_min"] <= POSTPRANDIAL_MIN) & (arrays["tsm_target_min"] >= 0)
        status["n_pre_calib"] = int(arrays["pre_calib"].sum())
        status["n_anchor_in_horizon"] = int(arrays["anchor_in_horizon"].sum())
        status["n_clean"] = int(clean.sum())
        status["n_postprandial_clean"] = int(pp.sum())
        status["phys_blind_rmse_clean"] = round(_rmse(arrays["mech_fore"][clean], arrays["y"][clean]), 3)
        status["phys_aware_rmse_clean"] = round(_rmse(arrays["mech_fore_aware"][clean], arrays["y"][clean]), 3)
        status["phys_blind_rmse_pp"] = round(_rmse(arrays["mech_fore"][pp], arrays["y"][pp]), 3)
        status["phys_aware_rmse_pp"] = round(_rmse(arrays["mech_fore_aware"][pp], arrays["y"][pp]), 3)

        save = dict(arrays)
        save["has"] = np.array([has["hr"], has["act"], has["sleep"], has["spo2"]], dtype=np.float32)
        save["patient_id"] = np.array(pid)
        save["diet_cols"] = np.array(DIET_COLS)
        save["act_cols"] = np.array(ACT_COLS)

        # Here I write to a temp name then rename, so an interrupted run can
        # never leave a half-written .npz that the resume check would trust
        tmp_path = out_path + ".tmp.npz"
        np.savez_compressed(tmp_path, **save)
        os.replace(tmp_path, out_path)
        status["status"] = "ok"
    except Exception as e:
        status["status"] = "error"
        status["error"] = f"{type(e).__name__}: {e} | {traceback.format_exc().splitlines()[-3:]}"
    finally:
        status["seconds"] = round(time.time() - t0, 1)
    return status


def _append_status_row(row):
    new_file = not os.path.exists(STATUS_CSV)
    with open(STATUS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=STATUS_FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="Build the CGMacros six-encoder window cache.")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 2)),
                    help="parallel worker processes (default: min(4, CPU count minus 2))")
    ap.add_argument("--subjects", type=str, default="",
                    help="comma-separated subject IDs to build, e.g. 3 or 3,5")
    ap.add_argument("--force", action="store_true", help="rebuild subjects that already have a cache file")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    wanted = {int(x) for x in args.subjects.split(",") if x.strip()} if args.subjects else None

    tasks, n_existing = [], 0
    for path in sorted(glob.glob(os.path.join(COHORT_DIR, "cgmacros_*_diet_features.csv"))):
        m = FILE_RE.match(os.path.basename(path))
        if not m:
            continue
        pid = int(m.group(1))
        if pid in EXCLUDE_SUBJECTS:
            continue
        if wanted is not None and pid not in wanted:
            continue
        out_path = os.path.join(OUT_DIR, f"subject_{pid}.npz")
        if os.path.exists(out_path) and not args.force:
            n_existing += 1
            continue
        tasks.append((pid, path, out_path))

    total = len(tasks)
    print(f"Subjects to build: {total} | already cached (skipped): {n_existing} | workers: {args.workers}", flush=True)
    if total == 0:
        return

    done, counts, t_start = 0, {}, time.time()
    with Pool(processes=args.workers) as pool:
        for st in pool.imap_unordered(process_subject, tasks, chunksize=1):
            done += 1
            counts[st["status"]] = counts.get(st["status"], 0) + 1
            _append_status_row(st)
            elapsed = time.time() - t_start
            eta_min = elapsed / done * (total - done) / 60
            print(f"[{done}/{total}] subject {st['subject']}: {st['status']} | "
                  f"windows={st['n_windows']} clean={st['n_clean']} pp={st['n_postprandial_clean']} | "
                  f"{st['seconds']}s | elapsed {elapsed / 60:.1f} min, ETA ~{eta_min:.0f} min", flush=True)
            if st["error"]:
                print(f"    note: {st['error']}", flush=True)

    print("\nDone. Status counts this run:", counts)
    print(f"Per-subject details: {STATUS_CSV}")


if __name__ == "__main__":
    freeze_support()
    main()