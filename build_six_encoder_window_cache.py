"""
build_six_encoder_window_cache.py
==========================================================
Builds a per-patient cache of six-encoder training windows for the whole
cohort, using the EXACT same GlycemicWindowDataset (and therefore the same
fixed simglucose warm-start engine) that train_patient1031.py uses. Nothing
in window_generator.py is duplicated or modified here.

Where it lives: D:\\FYP\\CODE (outer repo root), next to the other batch_*
scripts. Run it from D:\\FYP\\CODE.

What it writes (results/ is gitignored, so none of this is committed):
    results/six_encoder_cache/patient_<id>.npz   one file per patient
    results/six_encoder_cache/build_status.csv   one row per patient built

Each .npz has two splits, named "train" and "later", cut exactly the way
get_dataloader() cuts them (first train_days*288 rows vs the rest), and each
split is built separately with its own calibration points, as in
get_dataloader(). Keys per split (prefix "train_" or "later_"):
    hr (N,12)  act (N,12,2)  sleep (N,12)  spo2 (N,12)  glc (N,2)
    mech_curr (N,)  mech_fore (N,)  y (N,)  pre_calib (N,) bool
    has (4,) = [hr, act, sleep, spo2] availability for that split
Values are stored RAW, exactly as the generator produces them (no scaling).
cgm_window is not cached because HybridResidualTwin.forward() never reads it.
Samples stay in time order, so the "later" split can be cut into a
validation part and a test part by position.

Resumable: a patient whose .npz already exists is skipped (use --force to
rebuild). Files are written to a temp name and renamed, so an interrupted run
never leaves a half-written cache file behind.
"""

import os
import re
import csv
import sys
import time
import argparse
import traceback
from multiprocessing import Pool, freeze_support

import numpy as np
import pandas as pd

# Here I put this script's own folder on sys.path so that
# glycemic_twin.ml_layer.window_generator imports the same way it does in
# train_patient1031.py
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

COHORT_DIR = os.path.join("results", "cohort")
OUT_DIR = os.path.join("results", "six_encoder_cache")
STATUS_CSV = os.path.join(OUT_DIR, "build_status.csv")

# Here I keep the same outlier blocklist the LSTM pipeline uses (Patient 1027)
EXCLUDE_PATIENT_IDS = {1027}
# Here I match only the real merged files, so backup copies like
# patient_1031_merged_backup.csv are never picked up by accident
PATIENT_FILE_RE = re.compile(r"^patient_(\d+)_merged\.csv$")
STEPS_PER_DAY = 288  # 5-min grid

SEQ_KEYS = ("hr", "act", "sleep", "spo2")

STATUS_FIELDS = [
    "patient_id", "status", "rows", "n_train", "n_later",
    "has_hr", "has_act", "has_sleep", "has_spo2",
    "spo2_raw_nonnull_frac",
    "n_pre_calib_train", "n_pre_calib_later",
    "phys_rmse_train", "phys_rmse_later", "phys_rmse_later_postcalib",
    "n_dropped_nonfinite", "disabled_modalities", "later_note",
    "seconds", "error",
]


def _worker_init():
    # Here I pin each worker to one torch thread so N workers do not fight
    # over the same cores (simglucose itself is single-threaded Python)
    import torch
    torch.set_num_threads(1)


def _rmse(a, b):
    if len(a) == 0:
        return float("nan")
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sqrt(np.mean(d ** 2)))


def build_split(df_split, dataset_cls):
    """Runs the real GlycemicWindowDataset on one split and stacks its samples."""
    if len(df_split) == 0:
        return None, "empty_split"
    try:
        ds = dataset_cls(df_split)
    except (ValueError, KeyError) as e:
        return None, f"{type(e).__name__}: {e}"

    samples = ds.samples
    if len(samples) == 0:
        return None, "no_windows"

    arrays = {
        "hr": np.stack([s["hr_window"].numpy()[:, 0] for s in samples]).astype(np.float32),
        "act": np.stack([s["act_window"].numpy() for s in samples]).astype(np.float32),
        "sleep": np.stack([s["sleep_window"].numpy()[:, 0] for s in samples]).astype(np.float32),
        "spo2": np.stack([s["spo2_window"].numpy()[:, 0] for s in samples]).astype(np.float32),
        "glc": np.stack([s["glc_context"].numpy() for s in samples]).astype(np.float32),
        "mech_curr": np.array([s["g_mech_curr"].item() for s in samples], dtype=np.float32),
        "mech_fore": np.array([s["g_mech_fore"].item() for s in samples], dtype=np.float32),
        "y": np.array([s["y_target"].item() for s in samples], dtype=np.float32),
    }
    has = {
        "hr": float(samples[0]["has_hr"].item()),
        "act": float(samples[0]["has_act"].item()),
        "sleep": float(samples[0]["has_sleep"].item()),
        "spo2": float(samples[0]["has_spo2"].item()),
    }

    # Here I flag windows whose "physics" baseline is not simglucose output.
    # compute_simglucose_baseline() fills every step BEFORE the first
    # calibration point with glucose[0] (a flat real-CGM value), and the
    # `== 0` skip in GlycemicWindowDataset never fires for those steps. So
    # any window whose current or forecast baseline equals glucose[0] exactly
    # is treated as pre-calibration. This is an equality heuristic, not a
    # replay of the generator's loop, on purpose: I do not want a second copy
    # of that logic to drift out of sync.
    g0 = df_split["glucose_mg_dl"].values.astype(np.float32)[0]
    arrays["pre_calib"] = (arrays["mech_curr"] == g0) | (arrays["mech_fore"] == g0)
    return {"arrays": arrays, "has": has}, ""


def sanitize(split):
    """
    Here I make sure no NaN/inf reaches the model. A modality whose whole
    array is non-finite (e.g. an SpO2 column that exists but is 100% NaN) is
    switched OFF (zeros, has=0), which is honest: no real data. Remaining
    windows with any non-finite input are DROPPED and counted, not
    zero-filled, so no fake zeros get presented as real signal.
    """
    arrays, has = split["arrays"], split["has"]
    disabled = []
    for k in SEQ_KEYS:
        if has[k] and not np.isfinite(arrays[k]).any():
            arrays[k] = np.zeros_like(arrays[k])
            has[k] = 0.0
            disabled.append(k)

    n = len(arrays["y"])
    keep = np.ones(n, dtype=bool)
    for k in SEQ_KEYS + ("glc",):
        keep &= np.isfinite(arrays[k].reshape(n, -1)).all(axis=1)
    for k in ("mech_curr", "mech_fore", "y"):
        keep &= np.isfinite(arrays[k])

    n_dropped = int((~keep).sum())
    if n_dropped:
        arrays = {k: v[keep] for k, v in arrays.items()}
    return {"arrays": arrays, "has": has}, n_dropped, disabled


def _empty_split():
    return {
        "arrays": {
            "hr": np.zeros((0, 12), np.float32), "act": np.zeros((0, 12, 2), np.float32),
            "sleep": np.zeros((0, 12), np.float32), "spo2": np.zeros((0, 12), np.float32),
            "glc": np.zeros((0, 2), np.float32), "mech_curr": np.zeros((0,), np.float32),
            "mech_fore": np.zeros((0,), np.float32), "y": np.zeros((0,), np.float32),
            "pre_calib": np.zeros((0,), bool),
        },
        "has": {"hr": 0.0, "act": 0.0, "sleep": 0.0, "spo2": 0.0},
    }


def process_patient(task):
    pid, csv_path, out_path, train_days = task
    t0 = time.time()
    status = {k: "" for k in STATUS_FIELDS}
    status["patient_id"] = pid
    try:
        # Here I import inside the worker so each spawned process loads the
        # simglucose engine itself (required for Windows multiprocessing)
        from glycemic_twin.ml_layer.window_generator import GlycemicWindowDataset

        df = pd.read_csv(csv_path)
        status["rows"] = len(df)
        if "oxygen_saturation" in df.columns:
            # Here I record how much of SpO2 is real vs ffill/bfill'd later on
            status["spo2_raw_nonnull_frac"] = round(float(df["oxygen_saturation"].notna().mean()), 4)

        split_idx = train_days * STEPS_PER_DAY
        df_train = df.iloc[:split_idx].reset_index(drop=True)
        df_later = df.iloc[split_idx:].reset_index(drop=True)

        total_dropped = 0
        all_disabled = []
        splits = {}
        for name, df_split in (("train", df_train), ("later", df_later)):
            built, reason = build_split(df_split, GlycemicWindowDataset)
            if built is None:
                splits[name] = None
                if name == "later":
                    status["later_note"] = reason
                else:
                    status["error"] = f"train split failed: {reason}"
                continue
            built, n_dropped, disabled = sanitize(built)
            total_dropped += n_dropped
            all_disabled += [f"{name}:{d}" for d in disabled]
            splits[name] = built

        status["n_dropped_nonfinite"] = total_dropped
        status["disabled_modalities"] = ";".join(all_disabled)

        if splits["train"] is None:
            status["status"] = "error_no_train"
            return status

        for name in ("train", "later"):
            sp = splits[name]
            if sp is None:
                status[f"n_{name}"] = 0
                continue
            a = sp["arrays"]
            status[f"n_{name}"] = len(a["y"])
            status[f"n_pre_calib_{name}"] = int(a["pre_calib"].sum())
            status[f"phys_rmse_{name}"] = round(_rmse(a["mech_fore"], a["y"]), 3)
            if name == "later":
                post = ~a["pre_calib"]
                status["phys_rmse_later_postcalib"] = round(_rmse(a["mech_fore"][post], a["y"][post]), 3)
        h = splits["train"]["has"]
        status["has_hr"], status["has_act"] = int(h["hr"]), int(h["act"])
        status["has_sleep"], status["has_spo2"] = int(h["sleep"]), int(h["spo2"])

        save = {"patient_id": np.array(pid), "train_days": np.array(train_days)}
        for name in ("train", "later"):
            sp = splits[name] if splits[name] is not None else _empty_split()
            for k, v in sp["arrays"].items():
                save[f"{name}_{k}"] = v
            save[f"{name}_has"] = np.array(
                [sp["has"]["hr"], sp["has"]["act"], sp["has"]["sleep"], sp["has"]["spo2"]],
                dtype=np.float32,
            )

        # Here I write to a temp name then rename, so an interrupted run can
        # never leave a half-written .npz that the resume check would trust
        tmp_path = out_path + ".tmp.npz"
        np.savez_compressed(tmp_path, **save)
        os.replace(tmp_path, out_path)

        status["status"] = "ok" if splits["later"] is not None else "ok_no_later"
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
    ap = argparse.ArgumentParser(description="Build the six-encoder window cache for the cohort.")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                    help="parallel worker processes (default: CPU count minus 2)")
    ap.add_argument("--patients", type=str, default="",
                    help="comma-separated patient IDs to build, e.g. 1031 or 1031,1032")
    ap.add_argument("--limit", type=int, default=0, help="only build the first N patients (0 = all)")
    ap.add_argument("--train-days", type=int, default=8, help="same split as get_dataloader (default 8)")
    ap.add_argument("--force", action="store_true", help="rebuild patients that already have a cache file")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    wanted = {int(x) for x in args.patients.split(",") if x.strip()} if args.patients else None
    tasks, n_existing = [], 0
    for fname in sorted(os.listdir(COHORT_DIR)):
        m = PATIENT_FILE_RE.match(fname)
        if not m:
            continue
        pid = int(m.group(1))
        if pid in EXCLUDE_PATIENT_IDS:
            continue
        if wanted is not None and pid not in wanted:
            continue
        out_path = os.path.join(OUT_DIR, f"patient_{pid}.npz")
        if os.path.exists(out_path) and not args.force:
            n_existing += 1
            continue
        tasks.append((pid, os.path.join(COHORT_DIR, fname), out_path, args.train_days))
    if args.limit:
        tasks = tasks[: args.limit]

    total = len(tasks)
    print(f"Patients to build: {total} | already cached (skipped): {n_existing} | workers: {args.workers}", flush=True)
    if total == 0:
        return

    done, counts, t_start = 0, {}, time.time()
    with Pool(processes=args.workers, initializer=_worker_init) as pool:
        for st in pool.imap_unordered(process_patient, tasks, chunksize=1):
            done += 1
            counts[st["status"]] = counts.get(st["status"], 0) + 1
            _append_status_row(st)
            elapsed = time.time() - t_start
            eta_min = elapsed / done * (total - done) / 60
            print(f"[{done}/{total}] patient {st['patient_id']}: {st['status']} | "
                  f"train={st['n_train']} later={st['n_later']} | {st['seconds']}s | "
                  f"elapsed {elapsed / 60:.1f} min, ETA ~{eta_min:.0f} min", flush=True)
            if st["error"]:
                print(f"    note: {st['error']}", flush=True)

    print("\nDone. Status counts this run:", counts)
    print(f"Per-patient details: {STATUS_CSV}")


if __name__ == "__main__":
    freeze_support()
    main()