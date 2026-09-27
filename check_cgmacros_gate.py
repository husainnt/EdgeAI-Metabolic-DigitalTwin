"""
check_cgmacros_gate.py
==========================================================
Runs postprandial_gate_report() across one or more CGMacros subjects.
The gate: does injecting real logged meals into simglucose beat NOT
injecting them, in the hours after each meal. A subject failing this
is a real, reportable result (simglucose's generic digestion dynamics
not transferring cleanly to that person), not a bug to hide.

Usage:
    python check_cgmacros_gate.py 3          # one subject
    python check_cgmacros_gate.py             # all 13 T2D subjects
"""
import sys
import glob
import re
import pandas as pd
from cgmacros_meal_aware_physics import postprandial_gate_report

COHORT_DIR = "results/cgmacros_cohort"

# Here I keep this in sync with extract_cgmacros_diet_features.py's
# EXCLUDE_SUBJECTS -- the extraction script only skips REGENERATING an
# excluded subject's file, it does not delete a stale one left over from
# before the exclusion existed, so this is a second, defensive filter
EXCLUDE_SUBJECTS = {30}


def run_one(pid: int):
    path = f"{COHORT_DIR}/cgmacros_{pid:03d}_diet_features.csv"
    df = pd.read_csv(path)
    print(f"\n=== Subject {pid} ({len(df)} rows, "
          f"{int((df['time_since_last_meal_min']==0).sum())} meals) ===")
    try:
        return pid, postprandial_gate_report(df)
    except Exception as e:
        print(f"[!] Subject {pid} FAILED: {type(e).__name__}: {e}")
        return pid, None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pids = [int(sys.argv[1])]
    else:
        files = sorted(glob.glob(f"{COHORT_DIR}/cgmacros_*_diet_features.csv"))
        pids = [int(re.search(r"cgmacros_(\d+)_", f).group(1)) for f in files]
        skipped = [p for p in pids if p in EXCLUDE_SUBJECTS]
        if skipped:
            print(f"[i] Skipping stale excluded-subject file(s) for {skipped} -- "
                  f"delete these from {COHORT_DIR} to stop this warning.")
        pids = [p for p in pids if p not in EXCLUDE_SUBJECTS]

    results = [run_one(p) for p in pids]

    ok = [(p, r) for p, r in results if r is not None]
    print(f"\n{'='*60}")
    print(f"GATE SUMMARY: {len(ok)}/{len(pids)} subjects processed")
    n_pass = sum(1 for _, r in ok if r["passes"])
    print(f"  Passed (meal-aware beats meal-blind): {n_pass}/{len(ok)}")
    for p, r in ok:
        print(f"  Subject {p}: blind={r['rmse_blind']:.2f} aware={r['rmse_aware']:.2f} "
              f"n_meals={r['n_meals']} {'PASS' if r['passes'] else 'FAIL'}")
    print(f"{'='*60}")