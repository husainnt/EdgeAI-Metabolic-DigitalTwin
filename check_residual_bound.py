"""Checks whether the residual head's +/-80 mg/dL bound can even reach the target
on the meal-blind CGMacros baseline.

Usage (from D:\\FYP\\CODE):
    python check_residual_bound.py results\\cgmacros_window_cache

Here I measure |y - mech_fore| on clean windows. If a large share of windows sit
beyond the bound, the head saturates: even a perfect Diet encoder could not close
the gap, and the diet effect would be understated.
"""
import sys
from pathlib import Path

import numpy as np

# Here I take the cache directory from the command line
cache_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/cgmacros_window_cache")
BOUND = 80.0
POSTPRANDIAL_MIN = 180

print(f"{'subject':>7} {'n_clean':>8} {'n_pp':>6} {'|err|>80 all':>13} {'|err|>80 pp':>12} "
      f"{'median|err| pp':>15} {'mean signed pp':>15} {'best-case RMSE pp':>18}")
for path in sorted(cache_dir.glob("subject_*.npz"), key=lambda p: int(p.stem.split("_")[1])):
    d = np.load(path, allow_pickle=False)
    clean = ~(d["pre_calib"] | d["anchor_in_horizon"])
    if not clean.any():
        print(f"{int(d['patient_id']):>7}   (no clean windows)")
        continue
    err = d["y"] - d["mech_fore"]  # positive = physics under-predicts
    pp = clean & (d["tsm_target_min"] >= 0) & (d["tsm_target_min"] <= POSTPRANDIAL_MIN)

    # Here I compute the RMSE a PERFECT head would still leave if its output is
    # clipped to +/-BOUND, which is the floor the bound imposes on this baseline
    clipped_resid = np.clip(err, -BOUND, BOUND)
    floor = float(np.sqrt(np.mean((err[pp] - clipped_resid[pp]) ** 2))) if pp.any() else float("nan")
    print(f"{int(d['patient_id']):>7} {int(clean.sum()):>8} {int(pp.sum()):>6} "
          f"{float((np.abs(err[clean]) > BOUND).mean()):>13.3f} "
          f"{float((np.abs(err[pp]) > BOUND).mean()) if pp.any() else float('nan'):>12.3f} "
          f"{float(np.median(np.abs(err[pp]))) if pp.any() else float('nan'):>15.1f} "
          f"{float(err[pp].mean()) if pp.any() else float('nan'):>15.1f} {floor:>18.1f}")