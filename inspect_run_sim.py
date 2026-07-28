"""
Inspect D:\FYP\CODE\run_sim.py to see T2DPancreaticController implementation
"""

run_sim_path = r"D:\FYP\CODE\run_sim.py"

print(f"Reading {run_sim_path}...\n" + "=" * 70)
with open(run_sim_path, "r", encoding="utf-8") as f:
    code = f.read()

print(code)