"""
Sleep Modality Context-Control Evaluation (Patient 1031).
Evaluates whether REAL Wearable Sleep metrics provide genuine marginal predictive 
value over the Context-Only baseline (tsc + calibval).
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

LOOKBACK = 12  # 60 mins lookback
HORIZON = 6    # 30 mins forecast horizon

torch.manual_seed(42)
np.random.seed(42)


class ContextOnlyResidualMLP(nn.Module):
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


class SingleStreamResidualLSTM(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.encoder = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x_seq, tsc, calibval):
        x_seq = x_seq.unsqueeze(-1)
        _, (h_n, _) = self.encoder(x_seq)
        h_last = h_n.squeeze(0)
        combined = torch.cat([h_last, tsc.unsqueeze(-1), calibval.unsqueeze(-1)], dim=1)
        return self.fusion_head(combined).squeeze(-1)


class GenericDataset(Dataset):
    def __init__(self, seq_feat, tsc, calibval, y):
        self.seq_feat = torch.tensor(seq_feat, dtype=torch.float32) if seq_feat is not None else None
        self.tsc = torch.tensor(tsc, dtype=torch.float32)
        self.calibval = torch.tensor(calibval, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.tsc)

    def __getitem__(self, idx):
        if self.seq_feat is not None:
            return self.seq_feat[idx], self.tsc[idx], self.calibval[idx], self.y[idx]
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


def run_sleep_experiment(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[!] Dataset missing at {csv_path}. Run extract_and_merge_sleep_data.py first!")

    df = pd.read_csv(csv_path)
    calib_indices = extract_2x_daily_calibrations(df)

    glucose_cols = [c for c in df.columns if any(k in c.lower() for k in ['cgm', 'glucose', 'val', 'reading', 'mg/dl'])]
    hr_cols = [c for c in df.columns if any(k in c.lower() for k in ['hr', 'heart', 'bpm', 'rate'])]
    sleep_cols = [c for c in df.columns if 'sleep' in c.lower()]

    if not sleep_cols:
        raise ValueError("[!] No sleep column found! Ensure target_csv points to the merged sleep file.")

    glucose = df[glucose_cols[0]].values.astype(np.float32)
    hr_data = df[hr_cols[0]].values.astype(np.float32)
    sleep_data = df[sleep_cols[0]].values.astype(np.float32)

    active_sleep_hours = (sleep_data.sum() * 5.0) / 60.0
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f" • Sleep active windows: {int(sleep_data.sum())} slots ({active_sleep_hours:.1f} hours / {len(df)*5/60:.1f} total hours)")

    hr_series = pd.Series(hr_data)
    volatility_data = hr_series.rolling(window=6, min_periods=1).std().fillna(0).values.astype(np.float32)

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

    X_sleep, X_vol, y_real, y_sim, tsc_arr, calib_arr = [], [], [], [], [], []

    for i in range(LOOKBACK, num_samples - HORIZON):
        X_sleep.append(sleep_data[i - LOOKBACK : i])
        X_vol.append(volatility_data[i - LOOKBACK : i])
        y_real.append(glucose[i + HORIZON])
        y_sim.append(sim_baseline[i + HORIZON])
        tsc_arr.append(time_since_calib_series[i + HORIZON])
        calib_arr.append(calib_val_series[i + HORIZON])

    X_sleep, X_vol = np.array(X_sleep), np.array(X_vol)
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

    def normalize_feature(feat):
        tr, te = feat[:split], feat[split:]
        m, s = tr.mean(), tr.std() + 1e-6
        return (tr - m) / s, (te - m) / s

    X_slp_tr_n, X_slp_te_n = normalize_feature(X_sleep)
    X_vol_tr_n, X_vol_te_n = normalize_feature(X_vol)

    rmse_physics = np.sqrt(np.mean((y_real_te - y_sim_te) ** 2))

    print("\n" + "=" * 80)
    print("      REAL SLEEP TELEMETRY VS CONTEXT-ONLY CONTROL (PATIENT 1031)")
    print(f"      Mechanistic Physics Baseline RMSE: {rmse_physics:.2f} mg/dL")
    print("=" * 80)

    # 1. Context Only Control
    loader_ctx = DataLoader(GenericDataset(None, tsc_n_tr, cv_n_tr, y_tr_res), batch_size=32, shuffle=True)
    model_ctx = ContextOnlyResidualMLP()
    opt_ctx = torch.optim.AdamW(model_ctx.parameters(), lr=0.001, weight_decay=1e-4)
    sch_ctx = CosineAnnealingLR(opt_ctx, T_max=100, eta_min=1e-5)
    loss_fn = nn.MSELoss()

    model_ctx.train()
    for epoch in range(1, 101):
        for b_tsc, b_cv, b_y in loader_ctx:
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

    print(f" • Context-Only Control (tsc + calibval) | RMSE: {rmse_ctx:.2f} mg/dL | MAE: {mae_ctx:.2f} mg/dL | Gain: {gain_ctx:+.2f}%")

    # 2. Real Sleep + Context
    loader_slp = DataLoader(GenericDataset(X_slp_tr_n, tsc_n_tr, cv_n_tr, y_tr_res), batch_size=32, shuffle=True)
    model_slp = SingleStreamResidualLSTM(16)
    opt_slp = torch.optim.AdamW(model_slp.parameters(), lr=0.001, weight_decay=1e-4)
    sch_slp = CosineAnnealingLR(opt_slp, T_max=100, eta_min=1e-5)

    model_slp.train()
    for epoch in range(1, 101):
        for b_slp, b_tsc, b_cv, b_y in loader_slp:
            opt_slp.zero_grad()
            pred = model_slp(b_slp, b_tsc, b_cv)
            loss = loss_fn(pred, b_y)
            loss.backward()
            opt_slp.step()
        sch_slp.step()

    model_slp.eval()
    with torch.no_grad():
        t_slp_te = torch.tensor(X_slp_te_n, dtype=torch.float32)
        pred_res_slp = model_slp(t_slp_te, t_tsc_te, t_cv_te).numpy()

    rmse_slp = np.sqrt(np.mean((y_real_te - (y_sim_te + pred_res_slp)) ** 2))
    mae_slp = np.mean(np.abs(y_real_te - (y_sim_te + pred_res_slp)))
    gain_slp = ((rmse_physics - rmse_slp) / rmse_physics) * 100

    print(f" • Real Garmin Sleep + Context           | RMSE: {rmse_slp:.2f} mg/dL | MAE: {mae_slp:.2f} mg/dL | Gain: {gain_slp:+.2f}%")

    # 3. HR Volatility Reference
    loader_vol = DataLoader(GenericDataset(X_vol_tr_n, tsc_n_tr, cv_n_tr, y_tr_res), batch_size=32, shuffle=True)
    model_vol = SingleStreamResidualLSTM(16)
    opt_vol = torch.optim.AdamW(model_vol.parameters(), lr=0.001, weight_decay=1e-4)
    sch_vol = CosineAnnealingLR(opt_vol, T_max=100, eta_min=1e-5)

    model_vol.train()
    for epoch in range(1, 101):
        for b_vol, b_tsc, b_cv, b_y in loader_vol:
            opt_vol.zero_grad()
            pred = model_vol(b_vol, b_tsc, b_cv)
            loss = loss_fn(pred, b_y)
            loss.backward()
            opt_vol.step()
        sch_vol.step()

    model_vol.eval()
    with torch.no_grad():
        t_vol_te = torch.tensor(X_vol_te_n, dtype=torch.float32)
        pred_res_vol = model_vol(t_vol_te, t_tsc_te, t_cv_te).numpy()

    rmse_vol = np.sqrt(np.mean((y_real_te - (y_sim_te + pred_res_vol)) ** 2))
    mae_vol = np.mean(np.abs(y_real_te - (y_sim_te + pred_res_vol)))
    gain_vol = ((rmse_physics - rmse_vol) / rmse_physics) * 100

    print(f" • HR Volatility + Context              | RMSE: {rmse_vol:.2f} mg/dL | MAE: {mae_vol:.2f} mg/dL | Gain: {gain_vol:+.2f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Explicitly points to newly generated merged file containing real Garmin sleep data
    target_csv = "results/patient_1031_real_cgm_hr_steps_sleep.csv"
    run_sleep_experiment(target_csv)