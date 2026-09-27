"""
check_export.py
==========================================================
Sanity check: recomputes physics and hybrid RMSE directly from the
exported JSON and compares them against the already-validated post-fix
numbers for this patient (physics 49.54, hybrid 33.61-34.45 across seeds)
-- confirms the export pipeline is producing the same numbers the training
script itself reported, not a divergent/buggy path.

Usage:
    python check_export.py results/twin_visualization_export/patient_1031_val_demo.json
"""
import sys
import json

path = sys.argv[1] if len(sys.argv) > 1 else "results/twin_visualization_export/patient_1031_val_demo.json"
d = json.load(open(path))
f = d["frames"]

real = [x["glucose_real"] for x in f]
phys = [x["glucose_physics"] for x in f]
hyb = [x["glucose_hybrid"] for x in f]


def rmse(a, b):
    return (sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)) ** 0.5


print(f"n_frames: {len(f)}")
print(f"real range:    {min(real):.1f} - {max(real):.1f}")
print(f"physics range: {min(phys):.1f} - {max(phys):.1f}")
print(f"hybrid range:  {min(hyb):.1f} - {max(hyb):.1f}")
print(f"physics RMSE vs real: {rmse(phys, real):.2f}  (expected ~49.54)")
print(f"hybrid  RMSE vs real: {rmse(hyb, real):.2f}  (expected ~33.6-34.5)")