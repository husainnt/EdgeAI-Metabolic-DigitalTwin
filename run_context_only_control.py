"""
Context-Only Control Experiment.
Tests whether tsc (time since calibration) and calibval (last known glucose)
ALONE explain the residual correction — without any wearable sequence signal 
(HR, Steps, or Volatility) at all.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd

from run_selective_warmstart_simglucose_hybrid import (
    build_calibrated_t2d_patient,
    warm_start,
    simulate_open_loop,
    T2DPancreaticController
)

LOOKBACK = 12
HORIZON = 6

torch.manual_seed(42)
np.random.seed(42)


class ContextOnlyResidualMLP(nn.Module):
    """
    No sequence encoder at all — only tsc and calibval feed the prediction.
    Mirrors the fusion_head capacity of the other models for a fair comparison.
    """
    def __init__(self):
        super().__init__()
        self.fusion_head = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, tsc, calibval):
        combined = torch.cat([tsc.unsqueeze(-1), calibval.unsqueeze(-1)], dim=1)
        return self.fusion_head(combined).squeeze(-1)


class ContextOnlyDataset(Dataset):
    def __init__(self, tsc, calibval, y):
        self.tsc = torch.tensor(tsc, dtype=torch.float32)
        self.calibval = torch.tensor(calibval, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.tsc)

    def __getitem__(self, idx):
        return self.tsc[idx], self.calibval[idx], self.y[idx]


def extract_2x_daily_calibrations(df, window_tolerance_min=45):
    timestamps_naive = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    df['timestamp_naive'] = timestamps_naive
    dates = timestamps_naive.dt.date.unique()
    calib_indices = []

    for d in dates:
        day_df = df[df['timestamp_naive'].dt.date == d]
        t8 = pd.to_datetime(f"{d} 08:00:00")
        diffs8 = (day_df['timestamp_naive'] - t8).abs()
        if not diffs8.empty and diffs8.min() <= pd.Timedelta(minutes=window_tolerance_min):
            calib_indices.append(diffs8.idxmin())

        t20 = pd.to_datetime(f"{d} 20:00:00")
        diffs20 = (day_df['timestamp_naive'] - t20).abs()
        if not diffs20.empty and diffs20.min() <= pd.Timedelta(minutes=window_tolerance_min):
            calib_indices.append(diffs20.idxmin())

    return sorted(list(set(calib_indices)))


def run_context_only_control(df_path):
    if not os.path.exists(df_path):
        print(f"[!] Dataset missing at {df_path}")
        return

    df = pd.read_csv(df_path)
    calib_indices = extract_2x_daily_calibrations(df)
    
    glucose_cols = [c for c in df.columns if any(k in c.lower() for k in ['cgm', 'glucose', 'val', 'reading', 'mg/dl'])]
    glucose = df[glucose_cols[0]].values.astype(np.float32)

    print(f"Loaded {len(df)} rows from {df_path}")

    # 1. Generate simglucose ODE baseline
    patient = build_calibrated_t2d_patient()
    params = patient._params
    basal_rate_correct = params['u2ss'] * params['BW'] / 6000
    controller = T2DPancreaticController(gb=params['Gb'], basal_rate=basal_rate_correct)

    num_samples = len(df)
    sim_baseline = np.zeros(num_samples, dtype=np.float32)
    time_since_calib_series = np.zeros(num_samples, dtype=np.float32)
    calib_val_series = np.zeros(num_samples, dtype=np.float32)

    for k in range(len(calib_indices)):
        idx_start = calib_indices[k]
        idx_end = calib_indices[k + 1] if k + 1 < len(calib_indices) else num_samples
        n_steps = idx_end - idx_start
        g_calib = glucose[idx_start]

        patient.reset()
        warm_start(patient, g_calib)
        traj = simulate_open_loop(patient, controller, n_steps)

        for s_i, step_idx in enumerate(range(idx_start, idx_end)):
            sim_baseline[step_idx] = traj[s_i]
            time_since_calib_series[step_idx] = (s_i * 5.0) / 60.0
            calib_val_series[step_idx] = g_calib

    for i in range(calib_indices[0]):
        sim_baseline[i] = glucose[0]
        calib_val_series[i] = glucose[0]
        time_since_calib_series[i] = (i * 5.0) / 60.0

    # 2. Window sequences (context only)
    y_real, y_sim, tsc_arr, calib_arr = [], [], [], []

    for i in range(LOOKBACK, num_samples - HORIZON):
        y_real.append(glucose[i + HORIZON])
        y_sim.append(sim_baseline[i + HORIZON])
        tsc_arr.append(time_since_calib_series[i + HORIZON])
        calib_arr.append(calib_val_series[i + HORIZON])

    y_real, y_sim = np.array(y_real), np.array(y_sim)
    tsc_arr, calib_arr = np.array(tsc_arr), np.array(calib_arr)

    split = int(len(y_real) * 0.7)

    tsc_tr, tsc_te = tsc_arr[:split], tsc_arr[split:]
    cv_tr, cv_te = calib_arr[:split], calib_arr[split:]
    tsc_n_tr = (tsc_tr - tsc_tr.mean()) / (tsc_tr.std() + 1e-6)
    tsc_n_te = (tsc_te - tsc_tr.mean()) / (tsc_tr.std() + 1e-6)
    cv_n_tr = (cv_tr - cv_tr.mean()) / (cv_tr.std() + 1e-6)
    cv_n_te = (cv_te - cv_tr.mean()) / (cv_tr.std() + 1e-6)

    y_real_te, y_sim_te = y_real[split:], y_sim[split:]
    y_tr_res = y_real[:split] - y_sim[:split]

    rmse_physics = np.sqrt(np.mean((y_real_te - y_sim_te) ** 2))

    print("\n" + "=" * 80)
    print("      CONTEXT-ONLY CONTROL (tsc + calibval, NO sequence signal)")
    print(f"      Mechanistic Physics Baseline RMSE: {rmse_physics:.2f} mg/dL")
    print("=" * 80)

    loader = DataLoader(
        ContextOnlyDataset(tsc_n_tr, cv_n_tr, y_tr_res),
        batch_size=32, shuffle=True
    )

    model = ContextOnlyResidualMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, 101):
        for b_tsc, b_cv, b_y in loader:
            optimizer.zero_grad()
            pred = model(b_tsc, b_cv)
            loss = loss_fn(pred, b_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        t_tsc_te = torch.tensor(tsc_n_te, dtype=torch.float32)
        t_cv_te = torch.tensor(cv_n_te, dtype=torch.float32)
        pred_res = model(t_tsc_te, t_cv_te).numpy()

    pred_final = y_sim_te + pred_res
    rmse_hybrid = np.sqrt(np.mean((y_real_te - pred_final) ** 2))
    mae_hybrid = np.mean(np.abs(y_real_te - pred_final))
    gain_pct = ((rmse_physics - rmse_hybrid) / rmse_physics) * 100

    print(f" • Context Only (tsc + calibval)    | RMSE: {rmse_hybrid:.2f} mg/dL | MAE: {mae_hybrid:.2f} mg/dL | Improvement: {gain_pct:+.2f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_context_only_control("results/patient_1031_real_cgm_hr_steps.csv")