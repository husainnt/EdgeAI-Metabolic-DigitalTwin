"""
Patient 1205 Context-Only Control vs Raw HR Evaluation.
Isolates the contribution of calibration context (tsc + calibval) 
versus raw heart rate sequences for Patient 1205.
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


# --- 1. Model Definitions ---
class SelectiveWarmStartResidualLSTM(nn.Module):
    """Raw HR + Context (Standard Hybrid Twin)."""
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.hr_encoder = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x_hr, tsc, calibval):
        x_hr = x_hr.unsqueeze(-1)
        _, (h_n, _) = self.hr_encoder(x_hr)
        h_hr = h_n.squeeze(0)
        combined = torch.cat([h_hr, tsc.unsqueeze(-1), calibval.unsqueeze(-1)], dim=1)
        return self.fusion_head(combined).squeeze(-1)


class ContextOnlyResidualMLP(nn.Module):
    """Context Only Control (No HR sequence)."""
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


# --- 2. Dataset Utilities ---
class FullDataset(Dataset):
    def __init__(self, x_hr, tsc, calibval, y):
        self.x_hr = torch.tensor(x_hr, dtype=torch.float32)
        self.tsc = torch.tensor(tsc, dtype=torch.float32)
        self.calibval = torch.tensor(calibval, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x_hr)

    def __getitem__(self, idx):
        return self.x_hr[idx], self.tsc[idx], self.calibval[idx], self.y[idx]


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


def run_patient_1205_control(csv_path):
    if not os.path.exists(csv_path):
        print(f"[!] Dataset missing at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    calib_indices = extract_2x_daily_calibrations(df)

    glucose_cols = [c for c in df.columns if any(k in c.lower() for k in ['cgm', 'glucose', 'val', 'reading', 'mg/dl'])]
    hr_cols = [c for c in df.columns if any(k in c.lower() for k in ['hr', 'heart', 'bpm', 'rate'])]

    glucose = df[glucose_cols[0]].values.astype(np.float32)
    hr_data = df[hr_cols[0]].values.astype(np.float32)

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

    # 2. Window sequences
    X_hr, y_real, y_sim, tsc_arr, calib_arr = [], [], [], [], []

    for i in range(LOOKBACK, num_samples - HORIZON):
        X_hr.append(hr_data[i - LOOKBACK : i])
        y_real.append(glucose[i + HORIZON])
        y_sim.append(sim_baseline[i + HORIZON])
        tsc_arr.append(time_since_calib_series[i + HORIZON])
        calib_arr.append(calib_val_series[i + HORIZON])

    X_hr = np.array(X_hr)
    y_real, y_sim = np.array(y_real), np.array(y_sim)
    tsc_arr, calib_arr = np.array(tsc_arr), np.array(calib_arr)

    split = int(len(X_hr) * 0.7)

    # Context normalization
    tsc_tr, tsc_te = tsc_arr[:split], tsc_arr[split:]
    cv_tr, cv_te = calib_arr[:split], calib_arr[split:]
    tsc_n_tr, tsc_n_te = (tsc_tr - tsc_tr.mean()) / (tsc_tr.std() + 1e-6), (tsc_te - tsc_tr.mean()) / (tsc_tr.std() + 1e-6)
    cv_n_tr, cv_n_te = (cv_tr - cv_tr.mean()) / (cv_tr.std() + 1e-6), (cv_te - cv_tr.mean()) / (cv_tr.std() + 1e-6)

    # HR normalization
    hr_tr, hr_te = X_hr[:split], X_hr[split:]
    hr_m, hr_s = hr_tr.mean(), hr_tr.std() + 1e-6
    hr_n_tr, hr_n_te = (hr_tr - hr_m) / hr_s, (hr_te - hr_m) / hr_s

    y_real_te, y_sim_te = y_real[split:], y_sim[split:]
    y_tr_res = y_real[:split] - y_sim[:split]

    rmse_physics = np.sqrt(np.mean((y_real_te - y_sim_te) ** 2))

    print("\n" + "=" * 80)
    print("      PATIENT 1205: CONTEXT-ONLY VS RAW HR CONTROL EXPERIMENT")
    print(f"      Mechanistic Physics Baseline RMSE: {rmse_physics:.2f} mg/dL")
    print("=" * 80)

    # A) Run Context Only
    loader_ctx = DataLoader(
        FullDataset(hr_n_tr, tsc_n_tr, cv_n_tr, y_tr_res),
        batch_size=32, shuffle=True
    )
    model_ctx = ContextOnlyResidualMLP()
    opt_ctx = torch.optim.AdamW(model_ctx.parameters(), lr=0.001, weight_decay=1e-4)
    sch_ctx = CosineAnnealingLR(opt_ctx, T_max=100, eta_min=1e-5)
    loss_fn = nn.MSELoss()

    model_ctx.train()
    for epoch in range(1, 101):
        for _, b_tsc, b_cv, b_y in loader_ctx:
            opt_ctx.zero_grad()
            pred = model_ctx(b_tsc, b_cv)
            loss = loss_fn(pred, b_y)
            loss.backward()
            opt_ctx.step()
        sch_ctx.step()

    model_ctx.eval()
    with torch.no_grad():
        t_tsc_te = torch.tensor(tsc_n_te, dtype=torch.float32)
        t_cv_te = torch.tensor(cv_n_te, dtype=torch.float32)
        pred_res_ctx = model_ctx(t_tsc_te, t_cv_te).numpy()

    rmse_ctx = np.sqrt(np.mean((y_real_te - (y_sim_te + pred_res_ctx)) ** 2))
    mae_ctx = np.mean(np.abs(y_real_te - (y_sim_te + pred_res_ctx)))
    gain_ctx = ((rmse_physics - rmse_ctx) / rmse_physics) * 100

    print(f" • Context Only (tsc + calibval) | RMSE: {rmse_ctx:.2f} mg/dL | MAE: {mae_ctx:.2f} mg/dL | Gain: {gain_ctx:+.2f}%")

    # B) Run Raw HR + Context
    loader_hr = DataLoader(
        FullDataset(hr_n_tr, tsc_n_tr, cv_n_tr, y_tr_res),
        batch_size=32, shuffle=True
    )
    model_hr = SelectiveWarmStartResidualLSTM(hidden_dim=16)
    opt_hr = torch.optim.AdamW(model_hr.parameters(), lr=0.001, weight_decay=1e-4)
    sch_hr = CosineAnnealingLR(opt_hr, T_max=100, eta_min=1e-5)

    model_hr.train()
    for epoch in range(1, 101):
        for b_hr, b_tsc, b_cv, b_y in loader_hr:
            opt_hr.zero_grad()
            pred = model_hr(b_hr, b_tsc, b_cv)
            loss = loss_fn(pred, b_y)
            loss.backward()
            opt_hr.step()
        sch_hr.step()

    model_hr.eval()
    with torch.no_grad():
        t_hr_te = torch.tensor(hr_n_te, dtype=torch.float32)
        pred_res_hr = model_hr(t_hr_te, t_tsc_te, t_cv_te).numpy()

    rmse_hr = np.sqrt(np.mean((y_real_te - (y_sim_te + pred_res_hr)) ** 2))
    mae_hr = np.mean(np.abs(y_real_te - (y_sim_te + pred_res_hr)))
    gain_hr = ((rmse_physics - rmse_hr) / rmse_physics) * 100

    print(f" • Raw HR + Context              | RMSE: {rmse_hr:.2f} mg/dL | MAE: {mae_hr:.2f} mg/dL | Gain: {gain_hr:+.2f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_patient_1205_control("results/patient_1205_real_cgm_hr.csv")