"""Audit the CGMacros window cache after build_cgmacros_window_cache.py runs on real data.

Usage (from D:\\FYP\\CODE):
    python -u audit_cgmacros_cache.py results\\cgmacros_window_cache

Here I print the builder's own build_status.csv first, then re-derive the key
numbers from the npz files themselves, so a builder bug cannot hide behind its
own status report.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Here I take the cache directory from the command line
cache_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/cgmacros_window_cache")
POSTPRANDIAL_MIN = 180

status_path = cache_dir / "build_status.csv"
print("=" * 78)
print(f"build_status.csv: {status_path}")
if status_path.exists():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print(pd.read_csv(status_path).to_string(index=False))
else:
    print("MISSING - the builder did not write build_status.csv")

print("=" * 78)
rows = []
for npz_path in sorted(cache_dir.glob("subject_*.npz")):
    d = np.load(npz_path, allow_pickle=False)
    pid = int(d["patient_id"])
    print(f"\n--- {npz_path.name} ---")
    for key in d.files:
        arr = d[key]
        line = f"  {key:18s} shape={str(arr.shape):14s} dtype={str(arr.dtype):8s}"
        if np.issubdtype(arr.dtype, np.floating) and arr.size:
            line += (f" nan={int(np.isnan(arr).sum())} inf={int(np.isinf(arr).sum())}"
                     f" min={np.nanmin(arr):.3f} max={np.nanmax(arr):.3f}")
        print(line)

    # Here I re-derive the clean and postprandial subsets from the raw arrays
    clean = ~(d["pre_calib"] | d["anchor_in_horizon"])
    tsm = d["tsm_target_min"]
    pp = clean & (tsm >= 0) & (tsm <= POSTPRANDIAL_MIN)
    y, blind, aware = d["y"], d["mech_fore"], d["mech_fore_aware"]

    def rmse(m, x):
        return float(np.sqrt(np.mean((x[m] - y[m]) ** 2))) if m.any() else float("nan")

    # Here I check that meal injection actually changed the physics on real data;
    # a mean difference of exactly 0 would mean the meal path did nothing
    diff = float(np.mean(np.abs(aware - blind)))
    print(f"  mean |aware - blind| = {diff:.3f} mg/dL | mean(aware - blind) in pp windows = "
          f"{float(np.mean((aware - blind)[pp])) if pp.any() else float('nan'):.3f}")
    rows.append({"subject": pid, "windows": len(y), "clean": int(clean.sum()), "pp_clean": int(pp.sum()),
                 "blind_all": rmse(clean, blind), "aware_all": rmse(clean, aware),
                 "blind_pp": rmse(pp, blind), "aware_pp": rmse(pp, aware),
                 "dropped_by_flags": int((~clean).sum()), "capped_windows": int((d["macro_capped"] > 0).sum())})

print("\n" + "=" * 78)
if rows:
    summ = pd.DataFrame(rows).round(2)
    print("PER-SUBJECT SUMMARY (RMSE in mg/dL on clean windows; pp = within 3 h of a meal)")
    print(summ.to_string(index=False))
    print(f"\nsubjects cached: {len(summ)} | total clean windows: {int(summ['clean'].sum())} "
          f"| total postprandial clean: {int(summ['pp_clean'].sum())}")
    thin = summ[summ["pp_clean"] < 50]
    if len(thin):
        print(f"WARNING: subjects with <50 clean postprandial windows: {thin['subject'].tolist()}")