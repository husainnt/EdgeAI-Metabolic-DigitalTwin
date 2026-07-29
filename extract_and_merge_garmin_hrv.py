"""
Extracts Rolling Short-Term Heart Rate Volatility (Autonomic Tone Proxies)
from Patient 1031's 1-minute resolution Garmin heart rate data.

Features Computed:
1. hr_volatility_15m: Rolling standard deviation of 1-min HR over a 15-minute window (3 steps).
2. hr_volatility_30m: Rolling standard deviation of 1-min HR over a 30-minute window (6 steps).
3. hr_delta_5m: 5-minute heart rate gradient (rate of change in autonomic drive).
"""

import os
import pandas as pd
import numpy as np

INPUT_DATASET = "results/patient_1031_real_cgm_hr_steps.csv"
OUTPUT_DATASET = "results/patient_1031_real_cgm_hr_steps_hrv.csv"


def compute_hr_volatility_proxies(df):
    hr = df['heart_rate'].values
    
    # 1. 15-minute rolling HR volatility (3 x 5-min steps)
    hr_vol_15m = pd.Series(hr).rolling(window=3, min_periods=1).std().fillna(0.0).values
    
    # 2. 30-minute rolling HR volatility (6 x 5-min steps)
    hr_vol_30m = pd.Series(hr).rolling(window=6, min_periods=1).std().fillna(0.0).values
    
    # 3. 5-minute HR gradient (rate of change)
    hr_delta_5m = np.diff(hr, prepend=hr[0])

    df['hr_volatility_15m'] = hr_vol_15m
    df['hr_volatility_30m'] = hr_vol_30m
    df['hr_delta_5m'] = hr_delta_5m
    return df


def run_extraction():
    if not os.path.exists(INPUT_DATASET):
        raise FileNotFoundError(f"Input dataset not found at: {INPUT_DATASET}")

    df = pd.read_csv(INPUT_DATASET)
    print(f"[+] Loaded dataset with {len(df):,} records for Patient 1031.")

    df_out = compute_hr_volatility_proxies(df)

    os.makedirs("results", exist_ok=True)
    df_out.to_csv(OUTPUT_DATASET, index=False)

    print(f"[+] Successfully exported updated multimodal dataset to: {OUTPUT_DATASET}")
    print("\n--- Summary Statistics of Short-Term HR Volatility ---")
    print(f"    HR Volatility 15m (Mean ± Std): {df_out['hr_volatility_15m'].mean():.2f} ± {df_out['hr_volatility_15m'].std():.2f} bpm")
    print(f"    HR Volatility 30m (Mean ± Std): {df_out['hr_volatility_30m'].mean():.2f} ± {df_out['hr_volatility_30m'].std():.2f} bpm")
    print(f"    HR Delta 5m (Mean ± Std):      {df_out['hr_delta_5m'].mean():.2f} ± {df_out['hr_delta_5m'].std():.2f} bpm")
    print("\nFirst 5 rows preview:")
    print(df_out[['timestamp', 'glucose_mg_dl', 'heart_rate', 'steps', 'hr_volatility_15m', 'hr_volatility_30m']].head())


if __name__ == "__main__":
    run_extraction()