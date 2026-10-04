"""
analyze_physics_meal_gain.py
==========================================================
Quick follow-up on a finished train_cgmacros_diet_cv.py run, using only its saved
out-of-fold predictions (no retraining, runs in seconds).

Question: instead of a neural Diet encoder, is ONE learned scalar enough?
    prediction = pred_base + k * (meal_aware_physics - meal_blind_physics)
The meal-aware physics already encodes when and how big a meal response should be, but it
over-predicts on calibrated subjects, so k is expected to come out below 1. A single
parameter cannot memorise individual meals the way an encoder can.

k is fitted leave-one-subject-out: for each held-out subject it is estimated by least squares
on the OTHER subjects' windows, then applied to the held-out subject.

Usage (from D:\\FYP\\CODE):
    python analyze_physics_meal_gain.py results\\cgmacros_diet_cv\\cv_confirm
    python analyze_physics_meal_gain.py results\\cgmacros_diet_cv\\cv_confirm --base ctx_wear

Caveat: the other subjects' base predictions came from models that were trained on windows that
include the held-out subject, so this is an exploratory check with mild leakage, not a
confirmatory test.
"""
import glob
import os
import argparse

import numpy as np

POSTPRANDIAL_MIN = 180


def rmse(a, b):
    if len(a) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)))


def paired_stats(diff):
    # Here I use the same rule as train_cgmacros_diet_cv.py: negative mean = first arm better
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n < 2:
        return n, float("nan"), float("nan"), float("nan"), "too few subjects"
    mean, se = float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(n))
    verdict = ("IMPROVES (mean below 0 by more than 2 SE)" if mean + 2 * se < 0 else
               "WORSE (mean above 0 by more than 2 SE)" if mean - 2 * se > 0 else "not distinguishable from zero")
    return n, mean, se, float(np.mean(diff < 0)), verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--base", default="context", help="arm whose predictions get the physics gain added")
    ap.add_argument("--calibrated-rmse", type=float, default=60.0)
    ap.add_argument("--fit-on", choices=["all", "pp"], default="all",
                    help="fit k on all clean windows or only postprandial ones")
    args = ap.parse_args()

    folds = []
    for f in sorted(glob.glob(os.path.join(args.run_dir, "fold_*.npz"))):
        with np.load(f, allow_pickle=False) as d:
            folds.append({k: d[k] for k in d.files})
    if len(folds) < 3:
        raise SystemExit(f"[!] found {len(folds)} fold files in {args.run_dir}")
    if f"pred_{args.base}" not in folds[0]:
        raise SystemExit(f"[!] no pred_{args.base} in the fold files; arms present: "
                         f"{[k[5:] for k in folds[0] if k.startswith('pred_')]}")
    others = [k[5:] for k in folds[0] if k.startswith("pred_") and k[5:] != args.base]

    # Here I precompute the per-window residual of the base arm and the meal-physics signal
    for f in folds:
        f["r"] = f["y"] - f[f"pred_{args.base}"]
        f["x"] = f["mech_fore_aware"] - f["mech_fore"]
        f["pp"] = (f["tsm_target_min"] >= 0) & (f["tsm_target_min"] <= POSTPRANDIAL_MIN)

    rows, ks = [], []
    for i, f in enumerate(folds):
        xs, rs = [], []
        for j, g in enumerate(folds):
            if j == i:
                continue
            sel = g["pp"] if args.fit_on == "pp" else np.ones(len(g["y"]), bool)
            xs.append(g["x"][sel])
            rs.append(g["r"][sel])
        x, r = np.concatenate(xs), np.concatenate(rs)
        k = float((x * r).sum() / max((x * x).sum(), 1e-9))
        ks.append(k)
        p_gain = f[f"pred_{args.base}"] + k * f["x"]
        row = {"subject": int(f["pid"]), "k": k, "n_pp": int(f["pp"].sum())}
        row["phys_blind_all"] = rmse(f["mech_fore"], f["y"])
        series = {"base": f[f"pred_{args.base}"], "base+gain": p_gain}
        series.update({a: f[f"pred_{a}"] for a in others})
        for name, p in series.items():
            row[f"{name}_pp"] = rmse(p[f["pp"]], f["y"][f["pp"]])
        row["group"] = "calibrated" if row["phys_blind_all"] <= args.calibrated_rmse else "broken_baseline"
        rows.append(row)

    cols = ["base", "base+gain"] + others
    print("=" * 96)
    print(f"base arm = {args.base} | k fitted leave-one-subject-out on {args.fit_on} windows")
    print(f"k per held-out subject: min {min(ks):.2f}  median {np.median(ks):.2f}  max {max(ks):.2f}")
    print("=" * 96)
    print(f"{'subj':>5} {'group':<16} {'k':>6} " + " ".join(f"{c:>11}" for c in cols) + "   (postprandial RMSE, mg/dL)")
    for r in rows:
        print(f"{r['subject']:>5} {r['group']:<16} {r['k']:>6.2f} " + " ".join(f"{r[c + '_pp']:>11.2f}" for c in cols))

    print("\nPAIRED CONTRASTS, postprandial windows (negative = first arm better)")
    for grp in ("all_subjects", "calibrated", "broken_baseline"):
        sel = [r for r in rows if grp == "all_subjects" or r["group"] == grp]
        if not sel:
            continue
        pairs = [("base+gain", "base")] + [("base+gain", a) for a in others]
        for a, b in pairs:
            n, mean, se, frac, verdict = paired_stats([r[a + "_pp"] - r[b + "_pp"] for r in sel])
            la = a if a != "base" else args.base
            lb = b if b != "base" else args.base
            print(f"  [{grp:<15}] {a.replace('base', args.base) + ' - ' + lb:<34}: {mean:+7.2f} (SE {se:.2f}, n={n}, "
                  f"{100 * frac:.0f}% improved) -- {verdict}")
    print("=" * 96)


if __name__ == "__main__":
    main()