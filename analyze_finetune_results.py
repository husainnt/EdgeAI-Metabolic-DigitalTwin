# analyze_finetune_results.py
import re
import numpy as np
import pandas as pd

log_path = "finetune_full_log.txt"
pattern = re.compile(
    r"Patient\s+(\d+)\s+\|\s+Physics RMSE:\s*([\d\.]+)\s+\|\s+Zero-shot RMSE:\s*([\d\.]+)\s+\|\s+Fine-tuned RMSE:\s*([\d\.]+)\s+\|\s+Gain:\s*([+-]?[\d\.]+)%"
)

records = []
with open(log_path, "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            records.append({
                "patient_id": int(m.group(1)),
                "physics_rmse": float(m.group(2)),
                "zeroshot_rmse": float(m.group(3)),
                "finetuned_rmse": float(m.group(4)),
                "gain_pct": float(m.group(5)),
            })

df = pd.DataFrame(records)

print("=" * 75)
print(f"COHORT STATISTICAL BREAKDOWN (N = {len(df)})")
print("=" * 75)

# 1. Primary Central Tendencies
print("\n--- Central Tendency Metrics ---")
for col, name in [("physics_rmse", "Physics Baseline"), ("zeroshot_rmse", "Zero-Shot Backbone"), ("finetuned_rmse", "Fine-Tuned Twin")]:
    mean_val, std_val = df[col].mean(), df[col].std()
    med_val, q25, q75 = df[col].median(), df[col].quantile(0.25), df[col].quantile(0.75)
    print(f"{name:20s} | Mean: {mean_val:6.2f} ± {std_val:5.2f} mg/dL | Median: {med_val:5.2f} (IQR: {q25:5.2f} - {q75:5.2f})")

g_mean, g_std = df["gain_pct"].mean(), df["gain_pct"].std()
g_med, g_q25, g_q75 = df["gain_pct"].median(), df["gain_pct"].quantile(0.25), df["gain_pct"].quantile(0.75)
print(f"{'Cohort Error Reduction':20s} | Mean: {g_mean:6.2f}% ± {g_std:5.2f}% | Median: {g_med:5.2f}% (IQR: {g_q25:5.2f}% - {g_q75:5.2f}%)")

# 2. Gain Distribution
improved = df[df["gain_pct"] > 0]
degraded = df[df["gain_pct"] < 0]
neutral = df[df["gain_pct"] == 0]
print("\n--- Personalization Directionality ---")
print(f"Patients Improved (Gain > 0%):  {len(improved)} / {len(df)} ({len(improved)/len(df)*100:.1f}%)")
print(f"Patients Degraded (Gain < 0%):  {len(degraded)} / {len(df)} ({len(degraded)/len(df)*100:.1f}%)")
print(f"Substantial Gain (>= 40%):      {len(df[df['gain_pct'] >= 40])} / {len(df)} ({len(df[df['gain_pct'] >= 40])/len(df)*100:.1f}%)")

# 3. Degraded Patients Profile (The Negative Tail)
print("\n--- Failure Mode Analysis: Negative Gain Tail ---")
if len(degraded) > 0:
    print(degraded[["patient_id", "physics_rmse", "zeroshot_rmse", "finetuned_rmse", "gain_pct"]].to_string(index=False))
    print(f"\nMean Physics RMSE of Degraded Cohort: {degraded['physics_rmse'].mean():.2f} mg/dL")
    print(f"Mean Fine-Tuned RMSE of Degraded Cohort: {degraded['finetuned_rmse'].mean():.2f} mg/dL")

# 4. Extreme Drift Analysis (Physics > 150 mg/dL)
extreme = df[df["physics_rmse"] >= 150.0].sort_values(by="physics_rmse", ascending=False)
print(f"\n--- Extreme Drift Cohort (Physics RMSE >= 150 mg/dL, N = {len(extreme)}) ---")
print(extreme[["patient_id", "physics_rmse", "zeroshot_rmse", "finetuned_rmse", "gain_pct"]].to_string(index=False))
print("=" * 75)