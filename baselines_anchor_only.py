"""
baselines_anchor_only.py
==========================================================
Answers one question: how much of the six-encoder population result (physics 72.4 ->
47.7 mg/dL on the 88 held-out patients) can be reproduced by SIMPLE models that use the
same information?

It rebuilds the SAME test set as pretrain_six_encoder_population.py (md5 patient-ID hash,
buckets >= 85 are test patients, pre-calibration windows excluded, train + later windows
pooled) and compares, on those windows:
    physics            the simglucose forecast already in the cache
    train_mean         one constant, the mean glucose of the training patients
    zoh_anchor         hold the last calibration value
    decay_to_mean      last anchor relaxing toward the training mean with a fitted time constant
    ridge_anchor_phys  ridge on [anchor, hours since anchor, physics forecasts]   <- like-for-like
                       with the neural "Context-only" configuration
    ridge_all          the same plus simple wearable summaries                    <- like-for-like
                       with the neural "+ ALL real" configuration
    gbm_anchor_phys / gbm_all   the same two feature sets with gradient boosting (needs scikit-learn)
Everything is fitted on TRAIN patients (ridge strength is picked on VAL patients). Nothing is
fitted on test patients.

If you pass the neural model's per_patient_test.csv, it also prints paired per-patient
comparisons against the neural Context-only and + ALL real results.

Usage (from D:\\FYP\\CODE):
    python baselines_anchor_only.py
    python baselines_anchor_only.py --neural-csv results\\six_encoder_population\\curriculum\\per_patient_test.csv

Needs only numpy (pandas for the optional CSV). scikit-learn is optional.
Clock time is NOT stored in the cache, so a time-of-day baseline cannot be built from it.
"""
import os
import glob
import hashlib
import argparse

import numpy as np

FEATURES_ANCHOR = ["anchor", "tsc_h", "anchor_x_tsc", "tsc_sq"]
FEATURES_PHYS = ["mech_fore", "mech_curr", "mech_fore_minus_anchor"]
FEATURES_WEAR = ["hr_mean", "hr_std", "steps_log1p", "walk_frac", "sleep_mean", "spo2_mean",
                 "has_hr", "has_act", "has_sleep", "has_spo2"]


# ----------------------------------------------------------------------------
# Data: same patients, same windows as the neural run
# ----------------------------------------------------------------------------
def bucket(pid):
    # Here I use exactly the hash the population script uses, so the split is identical
    return int(hashlib.md5(str(pid).encode()).hexdigest(), 16) % 100


def load_features(cache_dir, keep_precalib=False):
    """One row per window. Wearable sequences are reduced to summaries while loading."""
    files = sorted(f for f in glob.glob(os.path.join(cache_dir, "patient_*.npz")) if not f.endswith(".tmp.npz"))
    if not files:
        raise SystemExit(f"[!] No patient_*.npz files in {cache_dir}")
    cols = {k: [] for k in ["pid", "y", "mech_curr", "mech_fore", "anchor", "tsc_h", "hr_mean", "hr_std",
                            "steps_log1p", "walk_frac", "sleep_mean", "spo2_mean",
                            "has_hr", "has_act", "has_sleep", "has_spo2"]}
    for f in files:
        with np.load(f) as d:
            pid = int(d["patient_id"])
            for split in ("train", "later"):
                n = len(d[f"{split}_y"])
                if n == 0:
                    continue
                keep = np.ones(n, bool) if keep_precalib else ~d[f"{split}_pre_calib"]
                m = int(keep.sum())
                if m == 0:
                    continue
                hr, act = d[f"{split}_hr"][keep], d[f"{split}_act"][keep]
                has = d[f"{split}_has"].astype(np.float32)   # [hr, act, sleep, spo2]
                glc = d[f"{split}_glc"][keep]
                cols["pid"].append(np.full(m, pid, np.int64))
                cols["y"].append(d[f"{split}_y"][keep])
                cols["mech_curr"].append(d[f"{split}_mech_curr"][keep])
                cols["mech_fore"].append(d[f"{split}_mech_fore"][keep])
                cols["tsc_h"].append(glc[:, 0])
                cols["anchor"].append(glc[:, 1])
                # Here I multiply every wearable summary by its availability flag, so a missing
                # sensor contributes zeros, the same convention the neural model's masks use
                cols["hr_mean"].append(hr.mean(axis=1) * has[0])
                cols["hr_std"].append(hr.std(axis=1) * has[0])
                cols["steps_log1p"].append(np.log1p(np.clip(act[..., 0].sum(axis=1), 0, None)) * has[1])
                cols["walk_frac"].append(act[..., 1].mean(axis=1) * has[1])
                cols["sleep_mean"].append(d[f"{split}_sleep"][keep].mean(axis=1) * has[2])
                cols["spo2_mean"].append(d[f"{split}_spo2"][keep].mean(axis=1) * has[3])
                for j, k in enumerate(["has_hr", "has_act", "has_sleep", "has_spo2"]):
                    cols[k].append(np.full(m, has[j], np.float32))
    D = {k: np.concatenate(v).astype(np.float64 if k == "pid" else np.float32) for k, v in cols.items()}
    D["pid"] = D["pid"].astype(np.int64)
    D["anchor_x_tsc"] = D["anchor"] * D["tsc_h"]
    D["tsc_sq"] = D["tsc_h"] ** 2
    D["mech_fore_minus_anchor"] = D["mech_fore"] - D["anchor"]
    up, inv = np.unique(D["pid"], return_inverse=True)
    b = np.array([bucket(p) for p in up])
    D["split"] = np.where(b < 70, 0, np.where(b < 85, 1, 2))[inv]   # 0 train, 1 val, 2 test
    return D


# ----------------------------------------------------------------------------
# Models (numpy)
# ----------------------------------------------------------------------------
def design(D, names, mask):
    return np.stack([D[n][mask] for n in names], axis=1).astype(np.float64)


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = (X - mu) / sd
    ym = y.mean()
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return lambda Xn: ((Xn - mu) / sd) @ w + ym


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p, np.float64) - np.asarray(y, np.float64)) ** 2)))


def per_patient(pred, y, pid):
    up, inv = np.unique(pid, return_inverse=True)
    se = (pred.astype(np.float64) - y.astype(np.float64)) ** 2
    return up, np.sqrt(np.bincount(inv, weights=se) / np.bincount(inv))


def paired(diff):
    # Here I use the same rule as the population script: mean +/- 2 SE around zero
    diff = np.asarray(diff, float)
    n = len(diff)
    mean, se = float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(n))
    return mean, se, float(np.mean(diff < 0)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.path.join("results", "six_encoder_cache"))
    ap.add_argument("--neural-csv", default=os.path.join("results", "six_encoder_population", "curriculum", "per_patient_test.csv"))
    ap.add_argument("--out-dir", default=os.path.join("results", "baselines"))
    ap.add_argument("--max-train-windows", type=int, default=400000, help="subsample for the boosting models")
    ap.add_argument("--keep-precalib", action="store_true")
    ap.add_argument("--expect-patients", type=int, default=88)
    ap.add_argument("--expect-windows", type=int, default=235883)
    ap.add_argument("--expect-physics", type=float, default=72.4267, help="pooled physics RMSE from results.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    D = load_features(args.cache_dir, args.keep_precalib)
    tr, va, te = D["split"] == 0, D["split"] == 1, D["split"] == 2
    print("=" * 100)
    print(f"cache: {args.cache_dir} | patients train/val/test = "
          f"{len(np.unique(D['pid'][tr]))}/{len(np.unique(D['pid'][va]))}/{len(np.unique(D['pid'][te]))} | "
          f"windows train/val/test = {tr.sum()}/{va.sum()}/{te.sum()}")
    phys_pooled = rmse(D["mech_fore"][te], D["y"][te])
    ok = (len(np.unique(D["pid"][te])) == args.expect_patients and int(te.sum()) == args.expect_windows
          and abs(phys_pooled - args.expect_physics) < 0.01)
    print(f"SANITY: test patients {len(np.unique(D['pid'][te]))} (expect {args.expect_patients}), test windows {int(te.sum())} "
          f"(expect {args.expect_windows}), pooled physics RMSE {phys_pooled:.4f} (expect {args.expect_physics}) -> "
          f"{'MATCHES the neural run' if ok else 'DOES NOT MATCH: the test set differs from the neural run, so comparisons are approximate'}")

    y_tr, y_te = D["y"][tr], D["y"][te]
    preds = {"physics": D["mech_fore"][te]}

    # ---- constant and anchor-only baselines ----
    m = float(y_tr.mean())
    preds["train_mean"] = np.full(te.sum(), m, np.float32)
    preds["zoh_anchor"] = D["anchor"][te]
    taus = [0.5, 1, 2, 3, 5, 8, 12, 24, 1e9]
    best_tau = min(taus, key=lambda t: rmse(m + (D["anchor"][tr] - m) * np.exp(-D["tsc_h"][tr] / t), y_tr))
    preds["decay_to_mean"] = m + (D["anchor"][te] - m) * np.exp(-D["tsc_h"][te] / best_tau)
    print(f"train mean glucose {m:.1f} mg/dL | decay time constant chosen on train: {best_tau:g} h")

    # ---- ridge, strength picked on the VAL patients ----
    feature_sets = {"anchor_phys": FEATURES_ANCHOR + FEATURES_PHYS,
                    "all": FEATURES_ANCHOR + FEATURES_PHYS + FEATURES_WEAR}
    for tag, names in feature_sets.items():
        Xtr, Xva, Xte = design(D, names, tr), design(D, names, va), design(D, names, te)
        best = min((1.0, 10.0, 100.0, 1000.0), key=lambda a: rmse(ridge_fit(Xtr, y_tr, a)(Xva), D["y"][va]))
        preds[f"ridge_{tag}"] = ridge_fit(Xtr, y_tr, best)(Xte)

    # ---- gradient boosting (optional) ----
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        idx = np.where(tr)[0]
        if len(idx) > args.max_train_windows:
            idx = rng.choice(idx, args.max_train_windows, replace=False)
        for tag, names in feature_sets.items():
            X = np.stack([D[n][idx] for n in names], axis=1)
            g = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                              random_state=args.seed).fit(X, D["y"][idx])
            preds[f"gbm_{tag}"] = g.predict(np.stack([D[n][te] for n in names], axis=1))
    except ImportError:
        print("[i] scikit-learn not installed: skipping the gradient-boosting baselines")

    # ---- summary ----
    pid_te = D["pid"][te]
    up, ph_pp = per_patient(preds["physics"], y_te, pid_te)
    rows = {}
    print("\n" + "=" * 100)
    print(f"{'model':<20} {'pooled':>8} {'mean/pt':>8} {'median/pt':>10} {'p90/pt':>8} {'better than physics':>20}")
    for name, p in preds.items():
        _, pp = per_patient(p, y_te, pid_te)
        rows[name] = pp
        print(f"{name:<20} {rmse(p, y_te):>8.2f} {pp.mean():>8.2f} {np.median(pp):>10.2f} {np.quantile(pp, .9):>8.2f} "
              f"{100 * float(np.mean(pp < ph_pp)):>19.1f}%")

    # ---- paired comparison with the neural results ----
    if args.neural_csv and os.path.exists(args.neural_csv):
        import pandas as pd
        nc = pd.read_csv(args.neural_csv).set_index("patient_id").reindex(up)
        gap = float(np.nanmax(np.abs(nc["physics_rmse"].values - ph_pp)))
        print(f"\nper-patient physics RMSE vs the neural CSV: max difference {gap:.3f} mg/dL "
              f"-> {'same windows' if gap < 0.01 else 'WINDOW SETS DIFFER, treat comparisons with caution'}")
        for ncol, like in (("Context-only", ("ridge_anchor_phys", "gbm_anchor_phys")), ("+ ALL real", ("ridge_all", "gbm_all"))):
            if ncol not in nc.columns:
                continue
            neural = nc[ncol].values
            print(f"\nNeural '{ncol}' (median per-patient RMSE {np.nanmedian(neural):.2f}). "
                  f"Paired difference = neural - baseline per patient (negative = neural better):")
            for b in ("train_mean", "zoh_anchor", "decay_to_mean") + like:
                if b not in rows:
                    continue
                mean, se, frac, n = paired(neural - rows[b])
                verdict = ("NEURAL BETTER" if mean + 2 * se < 0 else "NEURAL WORSE" if mean - 2 * se > 0
                           else "no distinguishable difference")
                print(f"  vs {b:<18}: {mean:+7.2f} (SE {se:.2f}, neural better in {100 * frac:.0f}% of {n} patients) -- {verdict}")
    else:
        print(f"\n[i] neural CSV not found at {args.neural_csv}; pass --neural-csv to get the paired comparison")

    # ---- error versus anchor age ----
    print("\n" + "=" * 100)
    print("POOLED RMSE BY ANCHOR AGE (hours since the last calibration), test windows")
    edges = [0, 1, 3, 6, 9, 13, 1e9]
    labels = ["<1h", "1-3h", "3-6h", "6-9h", "9-13h", ">=13h"]
    t = D["tsc_h"][te]
    show = [n for n in ("physics", "zoh_anchor", "ridge_anchor_phys", "gbm_anchor_phys") if n in preds]
    print(f"{'anchor age':<10} {'windows':>9} " + " ".join(f"{n:>18}" for n in show))
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        s = (t >= lo) & (t < hi)
        if s.sum() == 0:
            continue
        print(f"{lab:<10} {int(s.sum()):>9} " + " ".join(f"{rmse(preds[n][s], y_te[s]):>18.2f}" for n in show))
    print("=" * 100)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "baseline_per_patient.csv"), "w") as f:
        f.write("patient_id," + ",".join(rows) + "\n")
        for i, p in enumerate(up):
            f.write(f"{p}," + ",".join(f"{rows[k][i]:.3f}" for k in rows) + "\n")
    print(f"[SAVED] {os.path.join(args.out_dir, 'baseline_per_patient.csv')}")


if __name__ == "__main__":
    main()