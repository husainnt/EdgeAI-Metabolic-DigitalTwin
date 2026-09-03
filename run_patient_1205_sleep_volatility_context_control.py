"""
Sleep + HR Volatility vs Context-Only Control Evaluation (Patient 1205).
Confirms whether sleep and HR volatility generalize the same way raw HR did.
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
    def __init__(self):
        super().__init__()
        self.fusion_head = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, tsc, calibval):
        combined = torch.cat([tsc.unsqueeze(-1), calibval.unsqueeze(-1)], dim=1)
        return self.fusion_head(combined).squeeze(-1)


class SingleStreamResidualLSTM(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.encoder = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
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
    timestamps_naive = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df["timestamp_naive"] = timestamps_naive
    dates = timestamps_naive.dt.date.unique()
    calib_indices = []
    for d in dates:
        day_df = df[df["timestamp_naive"].dt.date == d]
        for target_hour in [8, 20]:
            t = pd.to_datetime(f"{d} {target_hour:02d}:00:00")
            diffs = (day_df["timestamp_naive"] - t).abs()
            if not diffs.empty and diffs.min() <= pd.Timedelta(minutes=window_tolerance_min):
                calib_indices.append(diffs.idxmin())
    return sorted(list(set(calib_indices)))


def run_eval(csv_path):
    df = pd.read_csv(csv_path)
    calib_indices = extract_2x_daily_calibrations(df)

    glucose = df["glucose_mg_dl"].values.astype(np.float32)
    hr = df["heart_rate"].values.astype(np.float32)
    sleep_data = df["sleep_status"].values.astype(np.float32)
    volatility = pd.Series(hr).rolling(window=6, min_periods=1).std().fillna(0).values.astype(np.float32)
    num_samples = len(df)

    print(f"[+] Loaded {num_samples} rows from {csv_path}")

    patient = build_calibrated_t2d_patient()
    params = patient._params
    basal_rate_correct = params["u2ss"] * params["BW"] / 6000
    controller = T2DPancreaticController(gb=params["Gb"], basal_rate=basal_rate_correct)

    sim_baseline = np.zeros(num_samples, dtype=np.float32)
    tsc_series = np.zeros(num_samples, dtype=np.float32)
    calib_series = np.zeros(num_samples, dtype=np.float32)

    segments = list(zip(calib_indices[:-1], calib_indices[1:]))
    if calib_indices[-1] < num_samples:
        segments.append((calib_indices[-1], num_samples))

    for idx_start, idx_end in segments:
        g_calib = glucose[idx_start]
        seg_len = idx_end - idx_start
        patient.reset()
        warm_start(patient, g_calib)
        traj = simulate_open_loop(patient, controller, seg_len)
        for s_i, step_idx in enumerate(range(idx_start, idx_end)):
            sim_baseline[step_idx] = traj[s_i]
            tsc_series[step_idx] = (s_i * 5.0) / 60.0
            calib_series[step_idx] = g_calib

    for i in range(calib_indices[0]):
        sim_baseline[i] = glucose[0]
        calib_series[i] = glucose[0]
        tsc_series[i] = (i * 5.0) / 60.0

    X_slp, X_vol, y_real, y_sim, tsc_arr, calib_arr = [], [], [], [], [], []
    for i in range(LOOKBACK, num_samples - HORIZON):
        if sim_baseline[i] == 0 or sim_baseline[i + HORIZON] == 0:
            continue
        X_slp.append(sleep_data[i - LOOKBACK:i])
        X_vol.append(volatility[i - LOOKBACK:i])
        y_real.append(glucose[i + HORIZON])
        y_sim.append(sim_baseline[i + HORIZON])
        tsc_arr.append(tsc_series[i + HORIZON])
        calib_arr.append(calib_series[i + HORIZON])

    X_slp, X_vol = np.array(X_slp), np.array(X_vol)
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

    def norm(feat):
        tr, te = feat[:split], feat[split:]
        m, s = tr.mean(), tr.std() + 1e-6
        return (tr - m) / s, (te - m) / s

    X_slp_tr_n, X_slp_te_n = norm(X_slp)
    X_vol_tr_n, X_vol_te_n = norm(X_vol)

    rmse_physics = np.sqrt(np.mean((y_real_te - y_sim_te) ** 2))

    print("\n" + "=" * 80)
    print("   PATIENT 1205: SLEEP + HR VOLATILITY VS CONTEXT-ONLY CONTROL")
    print(f"   Mechanistic Physics Baseline RMSE: {rmse_physics:.2f} mg/dL")
    print("=" * 80)

    def train_and_eval(name, model, seq_tr, seq_te):
        loader = DataLoader(GenericDataset(seq_tr, tsc_n_tr, cv_n_tr, y_tr_res), batch_size=32, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
        loss_fn = nn.MSELoss()
        model.train()
        for epoch in range(1, 101):
            for batch in loader:
                optimizer.zero_grad()
                if seq_tr is not None:
                    b_seq, b_tsc, b_cv, b_y = batch
                    pred = model(b_seq, b_tsc, b_cv)
                else:
                    b_tsc, b_cv, b_y = batch
                    pred = model(b_tsc, b_cv)
                loss = loss_fn(pred, b_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            t_tsc_te = torch.tensor(tsc_n_te, dtype=torch.float32)
            t_cv_te = torch.tensor(cv_n_te, dtype=torch.float32)
            if seq_te is not None:
                t_seq_te = torch.tensor(seq_te, dtype=torch.float32)
                pred_res = model(t_seq_te, t_tsc_te, t_cv_te).numpy()
            else:
                pred_res = model(t_tsc_te, t_cv_te).numpy()

        pred_final = y_sim_te + pred_res
        rmse = np.sqrt(np.mean((y_real_te - pred_final) ** 2))
        mae = np.mean(np.abs(y_real_te - pred_final))
        gain = ((rmse_physics - rmse) / rmse_physics) * 100
        print(f" \u2022 {name:<30} | RMSE: {rmse:.2f} mg/dL | MAE: {mae:.2f} | Gain: {gain:+.2f}%")
        return rmse

    train_and_eval("Context-Only Control", ContextOnlyResidualMLP(), None, None)
    train_and_eval("Real Sleep + Context", SingleStreamResidualLSTM(16), X_slp_tr_n, X_slp_te_n)
    train_and_eval("HR Volatility + Context", SingleStreamResidualLSTM(16), X_vol_tr_n, X_vol_te_n)

    print("=" * 80)


if __name__ == "__main__":
    run_eval("results/patient_1205_real_cgm_hr_sleep.csv")