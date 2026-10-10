# Here I summarize the 4-arm matrix: arm table, baselines, and paired per-patient contrasts over seeds
import csv
import json
import os
import sys
import numpy as np

# Here I read the results folder and seeds from the command line, with the Colab defaults
OUT = sys.argv[1] if len(sys.argv) > 1 else "/content/drive/MyDrive/FYP/results/matrix"
SEEDS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [42, 43, 44]
# Here I write the decision margin down once: a contrast only counts if it beats this many mg/dL
MARGIN = 0.5

# arm -> (description, config column that holds that arm's predictions)
ARMS = {
    "A": ("Transformer, physics base, all inputs", "+ ALL real"),
    "B": ("Transformer, anchor base, all inputs", "+ ALL real"),
    "C": ("LSTM-only (concat), anchor base, all inputs", "+ ALL real"),
    "D": ("LSTM-only (concat), anchor base, context only", "Context-only"),
    "E": ("Transformer, physics base, bound 300, all inputs", "+ ALL real"),
}


def read_run(arm, seed):
    # Here I load one run's results.json and its per-patient CSV, or return None if it has not finished
    d = os.path.join(OUT, f"{arm}_s{seed}")
    rj, cj = os.path.join(d, "results.json"), os.path.join(d, "per_patient_test.csv")
    if not (os.path.exists(rj) and os.path.exists(cj)):
        return None
    with open(rj) as f:
        res = json.load(f)
    with open(cj, newline="") as f:
        rows = list(csv.DictReader(f))
    return res, rows


runs = {}
for arm in ARMS:
    for seed in SEEDS:
        r = read_run(arm, seed)
        if r is not None:
            runs[(arm, seed)] = r
present = sorted({a for a, _ in runs})
if not present:
    raise SystemExit(f"No finished runs found in {OUT}")
print("[+] Finished runs: " + ", ".join(f"{a}x{sum(1 for (x, _) in runs if x == a)}" for a in present))

# Here I check that every run scored the same test patients, otherwise paired contrasts are meaningless
first_key = next(iter(runs))
ref_pids = [r["patient_id"] for r in runs[first_key][1]]
for k, (_, rows) in runs.items():
    assert [r["patient_id"] for r in rows] == ref_pids, f"test patients differ in run {k}"
res0 = runs[first_key][0]
print(f"[+] Test set: {res0['n_test_patients']} unseen patients, {res0['n_test_windows']:,} windows (look-ahead windows removed)")

# ---------------------------------------------------------------- baselines
print("\n" + "=" * 78)
print("BASELINES (same test windows, fit on train patients only)")
print("=" * 78)
print(f"{'Baseline':<26}{'pooled RMSE':>12}{'MAE':>9}{'median patient RMSE':>22}")
for name, v in res0["baselines"].items():
    print(f"{name:<26}{v['rmse']:>12.2f}{v['mae']:>9.2f}{v['median_patient_rmse']:>22.2f}")
print(f"anchor-decay fit: ybar {res0['anchor_fit']['ybar']:.2f} mg/dL, tau {res0['anchor_fit']['tau_hours']:g} h")


def pt_vec(arm, seed):
    # Here I return per-patient RMSE of an arm's predictions for one seed
    res, rows = runs[(arm, seed)]
    col = ARMS[arm][1]
    return np.array([float(r[col]) for r in rows])


def base_vec(name):
    # Here I return per-patient RMSE of a baseline (identical across seeds, so I read it from the first run)
    return np.array([float(r[f"base_{name}"]) for r in runs[first_key][1]])


def pooled(arm, seed):
    res, _ = runs[(arm, seed)]
    return res["results"][ARMS[arm][1]]["rmse"]


# ---------------------------------------------------------------- arm table
print("\n" + "=" * 100)
print("ARMS (mean +/- SD over seeds)")
print("=" * 100)
print(f"{'Arm':<4}{'Description':<50}{'pooled RMSE':>16}{'MAE':>8}{'median pt':>11}{'seeds':>7}")
for arm in present:
    seeds_here = [s for s in SEEDS if (arm, s) in runs]
    p = [pooled(arm, s) for s in seeds_here]
    mae = [runs[(arm, s)][0]["results"][ARMS[arm][1]]["mae"] for s in seeds_here]
    med = [np.median(pt_vec(arm, s)) for s in seeds_here]
    sd = np.std(p, ddof=1) if len(p) > 1 else float("nan")
    print(f"{arm:<4}{ARMS[arm][0]:<50}{np.mean(p):>10.2f} +/- {sd:<4.2f}{np.mean(mae):>8.2f}{np.mean(med):>11.2f}{len(p):>7}")
for ref in ("anchor_decay", "ridge_all"):
    v = res0["baselines"][ref]
    print(f"{'-':<4}{'baseline: ' + ref:<50}{v['rmse']:>10.2f}{'':>9}{v['mae']:>8.2f}{v['median_patient_rmse']:>11.2f}")


# ---------------------------------------------------------------- paired contrasts
def vec(label, seed):
    if label.startswith("base:"):
        return base_vec(label[5:])
    return pt_vec(label, seed)


def verdict(per_seed_means, se):
    m = float(np.mean(per_seed_means))
    same_sign = all(x < 0 for x in per_seed_means) or all(x > 0 for x in per_seed_means)
    if m < -MARGIN and same_sign and m + 2 * se < 0:
        return "BETTER, meets the decision rule"
    if m > MARGIN and same_sign and m - 2 * se > 0:
        return "WORSE, meets the decision rule"
    return "not distinguishable from zero / below the margin"


CONTRASTS = [
    ("base:anchor_decay", "A", "physics-base Transformer vs anchor-decay"),
    ("base:anchor_decay", "B", "anchor-base Transformer vs anchor-decay"),
    ("base:anchor_decay", "C", "LSTM-only vs anchor-decay"),
    ("base:ridge_all", "C", "LSTM-only vs ridge (all inputs)"),
    ("A", "B", "BASE effect: physics -> anchor (Transformer)"),
    ("B", "C", "FUSION effect: Transformer -> LSTM-only (anchor base)"),
    ("D", "C", "WEARABLE effect: context only -> all inputs (LSTM-only)"),
    ("A", "E", "CAP effect: bound 80 -> 300 (Transformer, physics base)"),
]
print("\n" + "=" * 100)
print(f"PAIRED PER-PATIENT CONTRASTS (second minus first, negative = second is better; decision margin {MARGIN} mg/dL)")
print("=" * 100)
for first, second, label in CONTRASTS:
    arms_needed = [x for x in (first, second) if not x.startswith("base:")]
    seeds_ok = [s for s in SEEDS if all((a, s) in runs for a in arms_needed)]
    if not seeds_ok:
        continue
    means, ses, imps = [], [], []
    for s in seeds_ok:
        dd = vec(second, s) - vec(first, s)
        means.append(float(dd.mean()))
        ses.append(float(dd.std(ddof=1) / np.sqrt(len(dd))))
        imps.append(float((dd < 0).mean() * 100))
    print(f"{label}")
    print(f"   {first} -> {second}: mean dRMSE {np.mean(means):+.3f} mg/dL | seed SD "
          f"{(np.std(means, ddof=1) if len(means) > 1 else float('nan')):.3f} | patient SE {np.mean(ses):.3f} | "
          f"improved {np.mean(imps):.0f}% | per seed {np.round(means, 3).tolist()}")
    print(f"   verdict: {verdict(means, np.mean(ses))}\n")