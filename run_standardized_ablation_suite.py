"""
Standardized Feature Ablation Suite (Apples-to-Apples Evaluation).
Evaluates HR alone, Steps alone, HR Volatility alone, Early Fusion, and Gated Late Fusion
under the exact single-target (+30 min) pipeline architecture on Patient 1031.
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


# --- 1. PyTorch Models for Feature Modalities ---

class SingleStreamResidualLSTM(nn.Module):
    """Standard single-modality LSTM (HR, Steps, or Volatility)."""
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


class EarlyFusionResidualLSTM(nn.Module):
    """Naive Early Fusion: Concatenates 2 sequence features into 2D input LSTM."""
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.encoder = nn.LSTM(input_size=2, hidden_size=hidden_dim, batch_first=True)
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x_feat1, x_feat2, tsc, calibval):
        x_dual = torch.stack([x_feat1, x_feat2], dim=-1)  # (batch, seq, 2)
        _, (h_n, _) = self.encoder(x_dual)
        h_last = h_n.squeeze(0)
        combined = torch.cat([h_last, tsc.unsqueeze(-1), calibval.unsqueeze(-1)], dim=1)
        return self.fusion_head(combined).squeeze(-1)


class GatedLateFusionResidualLSTM(nn.Module):
    """Gated Late Fusion: Separate LSTMs for HR & Steps with dynamic sigmoid gating."""
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.hr_encoder = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.step_encoder = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x_hr, x_steps, tsc, calibval):
        x_hr_in = x_hr.unsqueeze(-1)
        x_step_in = x_steps.unsqueeze(-1)
        
        _, (h_hr, _) = self.hr_encoder(x_hr_in)
        _, (h_step, _) = self.step_encoder(x_step_in)
        
        h_hr_last = h_hr.squeeze(0)
        h_step_last = h_step.squeeze(0)
        
        g = self.gate(torch.cat([h_hr_last, h_step_last], dim=1))
        h_fused = g * h_hr_last + (1.0 - g) * h_step_last
        
        combined = torch.cat([h_fused, tsc.unsqueeze(-1), calibval.unsqueeze(-1)], dim=1)
        return self.fusion_head(combined).squeeze(-1)


# --- 2. Dataset Utilities ---
class GenericAblationDataset(Dataset):
    def __init__(self, f1, f2, tsc, calibval, y):
        self.f1 = torch.tensor(f1, dtype=torch.float32)
        self.f2 = torch.tensor(f2, dtype=torch.float32) if f2 is not None else self.f1
        self.tsc = torch.tensor(tsc, dtype=torch.float32)
        self.calibval = torch.tensor(calibval, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.f1)

    def __getitem__(self, idx):
        return self.f1[idx], self.f2[idx], self.tsc[idx], self.calibval[idx], self.y[idx]


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


def run_ablation_experiment(df_path):
    if not os.path.exists(df_path):
        print(f"[!] Dataset missing at {df_path}")
        return

    df = pd.read_csv(df_path)
    calib_indices = extract_2x_daily_calibrations(df)

    glucose_cols = [c for c in df.columns if any(k in c.lower() for k in ['cgm', 'glucose', 'val', 'reading', 'mg/dl'])]
    hr_cols = [c for c in df.columns if any(k in c.lower() for k in ['hr', 'heart', 'bpm', 'rate'])]
    step_cols = [c for c in df.columns if 'step' in c.lower()]

    if not step_cols:
        raise ValueError(f"No step column found in {df_path}! Ensure you pass a CSV containing Garmin step data.")

    glucose = df[glucose_cols[0]].values.astype(np.float32)
    hr_data = df[hr_cols[0]].values.astype(np.float32)
    steps_data = df[step_cols[0]].values.astype(np.float32)

    print(f"Loaded {len(df)} rows from {df_path}")
    print(f" • Glucose Mean: {glucose.mean():.1f} mg/dL")
    print(f" • HR Mean:      {hr_data.mean():.1f} bpm")
    print(f" • Steps Sum:    {steps_data.sum():.0f} total steps")

    # Compute 30-min rolling HR volatility (std dev)
    hr_series = pd.Series(hr_data)
    volatility_data = hr_series.rolling(window=6, min_periods=1).std().fillna(0).values.astype(np.float32)

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
    X_hr, X_steps, X_vol, y_real, y_sim, tsc_arr, calib_arr = [], [], [], [], [], [], []

    for i in range(LOOKBACK, num_samples - HORIZON):
        X_hr.append(hr_data[i - LOOKBACK : i])
        X_steps.append(steps_data[i - LOOKBACK : i])
        X_vol.append(volatility_data[i - LOOKBACK : i])
        y_real.append(glucose[i + HORIZON])
        y_sim.append(sim_baseline[i + HORIZON])
        tsc_arr.append(time_since_calib_series[i + HORIZON])
        calib_arr.append(calib_val_series[i + HORIZON])

    X_hr, X_steps, X_vol = np.array(X_hr), np.array(X_steps), np.array(X_vol)
    y_real, y_sim = np.array(y_real), np.array(y_sim)
    tsc_arr, calib_arr = np.array(tsc_arr), np.array(calib_arr)

    split = int(len(X_hr) * 0.7)

    # Context normalization
    tsc_tr, tsc_te = tsc_arr[:split], tsc_arr[split:]
    cv_tr, cv_te = calib_arr[:split], calib_arr[split:]
    tsc_n_tr, tsc_n_te = (tsc_tr - tsc_tr.mean()) / (tsc_tr.std() + 1e-6), (tsc_te - tsc_tr.mean()) / (tsc_tr.std() + 1e-6)
    cv_n_tr, cv_n_te = (cv_tr - cv_tr.mean()) / (cv_tr.std() + 1e-6), (cv_te - cv_tr.mean()) / (cv_tr.std() + 1e-6)

    y_real_te, y_sim_te = y_real[split:], y_sim[split:]
    y_tr_res = y_real[:split] - y_sim[:split]

    def normalize_feature(feat):
        tr, te = feat[:split], feat[split:]
        m, s = tr.mean(), tr.std() + 1e-6
        return (tr - m) / s, (te - m) / s

    X_hr_tr_n, X_hr_te_n = normalize_feature(X_hr)
    X_stp_tr_n, X_stp_te_n = normalize_feature(X_steps)
    X_vol_tr_n, X_vol_te_n = normalize_feature(X_vol)

    rmse_physics = np.sqrt(np.mean((y_real_te - y_sim_te) ** 2))

    print("\n" + "=" * 80)
    print("      STANDARDIZED FEATURE ABLATION SUITE (PATIENT 1031)")
    print(f"      Mechanistic Physics Baseline RMSE: {rmse_physics:.2f} mg/dL")
    print("=" * 80)

    experiments = [
        ("Raw HR Alone", SingleStreamResidualLSTM(16), X_hr_tr_n, X_hr_te_n, None, None),
        ("Steps Alone", SingleStreamResidualLSTM(16), X_stp_tr_n, X_stp_te_n, None, None),
        ("HR Volatility Alone", SingleStreamResidualLSTM(16), X_vol_tr_n, X_vol_te_n, None, None),
        ("Early Fusion (HR + Steps)", EarlyFusionResidualLSTM(16), X_hr_tr_n, X_hr_te_n, X_stp_tr_n, X_stp_te_n),
        ("Gated Late Fusion (HR + Steps)", GatedLateFusionResidualLSTM(16), X_hr_tr_n, X_hr_te_n, X_stp_tr_n, X_stp_te_n),
    ]

    for name, model, f1_tr, f1_te, f2_tr, f2_te in experiments:
        loader = DataLoader(
            GenericAblationDataset(f1_tr, f2_tr, tsc_n_tr, cv_n_tr, y_tr_res),
            batch_size=32, shuffle=True
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
        loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(1, 101):
            for b_f1, b_f2, b_tsc, b_cv, b_y in loader:
                optimizer.zero_grad()
                if isinstance(model, SingleStreamResidualLSTM):
                    pred = model(b_f1, b_tsc, b_cv)
                elif isinstance(model, (EarlyFusionResidualLSTM, GatedLateFusionResidualLSTM)):
                    pred = model(b_f1, b_f2, b_tsc, b_cv)
                loss = loss_fn(pred, b_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            t_f1_te = torch.tensor(f1_te, dtype=torch.float32)
            t_f2_te = torch.tensor(f2_te, dtype=torch.float32) if f2_te is not None else None
            t_tsc_te = torch.tensor(tsc_n_te, dtype=torch.float32)
            t_cv_te = torch.tensor(cv_n_te, dtype=torch.float32)

            if isinstance(model, SingleStreamResidualLSTM):
                pred_res = model(t_f1_te, t_tsc_te, t_cv_te).numpy()
            elif isinstance(model, (EarlyFusionResidualLSTM, GatedLateFusionResidualLSTM)):
                pred_res = model(t_f1_te, t_f2_te, t_tsc_te, t_cv_te).numpy()

        pred_final = y_sim_te + pred_res
        rmse_hybrid = np.sqrt(np.mean((y_real_te - pred_final) ** 2))
        mae_hybrid = np.mean(np.abs(y_real_te - pred_final))
        gain_pct = ((rmse_physics - rmse_hybrid) / rmse_physics) * 100

        print(f" • {name:<32} | RMSE: {rmse_hybrid:.2f} mg/dL | MAE: {mae_hybrid:.2f} mg/dL | Improvement: {gain_pct:+.2f}%")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Corrected path to file containing glucose, heart rate, AND steps
    target_csv = "results/patient_1031_real_cgm_hr_steps.csv"
    if not os.path.exists(target_csv):
        # Fallback check if path differs slightly
        target_csv = "results/patient_1031_real_cgm_with_hr.csv"
        
    run_ablation_experiment(target_csv)