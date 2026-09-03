"""
Standardized Gb Personalization Ablation (Apples-to-Apples Pipeline).
Uses the exact SelectiveWarmStartResidualLSTM architecture and mini-batch AdamW
training loop from the headline 33.61 mg/dL experiment.
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


# --- 1. Exact Headline Model Architecture ---
class SelectiveWarmStartResidualLSTM(nn.Module):
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


class SelectiveWarmStartDataset(Dataset):
    def __init__(self, X_hr, tsc, calibval, y):
        self.X_hr = torch.tensor(X_hr, dtype=torch.float32)
        self.tsc = torch.tensor(tsc, dtype=torch.float32)
        self.calibval = torch.tensor(calibval, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X_hr)

    def __getitem__(self, idx):
        return self.X_hr[idx], self.tsc[idx], self.calibval[idx], self.y[idx]


# --- 2. Calibration Extraction ---
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


# --- 3. Dataset Generation ---
def generate_dataset_with_gb(df, calib_indices, custom_gb=None):
    glucose = df['glucose_mg_dl'].values
    hr = df['heart_rate'].values
    num_samples = len(df)

    sim_baseline = np.zeros(num_samples, dtype=np.float32)
    time_since_calib_series = np.zeros(num_samples, dtype=np.float32)
    calib_val_series = np.zeros(num_samples, dtype=np.float32)

    patient = build_calibrated_t2d_patient()
    if custom_gb is not None:
        patient._params['Gb'] = float(custom_gb)

    params = patient._params
    basal_rate_correct = params['u2ss'] * params['BW'] / 6000
    controller = T2DPancreaticController(gb=params['Gb'], basal_rate=basal_rate_correct)

    for k in range(len(calib_indices)):
        idx_start = calib_indices[k]
        idx_end = calib_indices[k + 1] if k + 1 < len(calib_indices) else len(glucose)
        n_steps = idx_end - idx_start
        g_calib = glucose[idx_start]

        patient.reset()
        if custom_gb is not None:
            patient._params['Gb'] = float(custom_gb)

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

    X_hr_seq, y_real, y_sim, tsc_arr, calib_arr = [], [], [], [], []

    for i in range(LOOKBACK, num_samples - HORIZON):
        X_hr_seq.append(hr[i - LOOKBACK : i])
        y_real.append(glucose[i + HORIZON])
        y_sim.append(sim_baseline[i + HORIZON])
        tsc_arr.append(time_since_calib_series[i + HORIZON])
        calib_arr.append(calib_val_series[i + HORIZON])

    return (np.array(X_hr_seq, dtype=np.float32),
            np.array(y_real, dtype=np.float32),
            np.array(y_sim, dtype=np.float32),
            np.array(tsc_arr, dtype=np.float32),
            np.array(calib_arr, dtype=np.float32))


# --- 4. Evaluation Engine ---
def evaluate_patient_gb(patient_id, csv_path):
    if not os.path.exists(csv_path):
        print(f"[!] File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    calib_indices = extract_2x_daily_calibrations(df)

    split_idx = int(len(df) * 0.7)
    train_mean_glucose = float(df['glucose_mg_dl'].iloc[:split_idx].mean())

    print("\n" + "=" * 75)
    print(f"   STANDARDIZED Gb ABLATION: PATIENT {patient_id}")
    print(f"   Train Window Gb: {train_mean_glucose:.1f} mg/dL | Template Gb: 138.6 mg/dL")
    print("=" * 75)

    results = {}

    for mode in ["default_gb", "personalized_gb"]:
        custom_gb = None if mode == "default_gb" else train_mean_glucose

        X_hr, y_real, y_sim, tsc, calib_val = generate_dataset_with_gb(df, calib_indices, custom_gb=custom_gb)

        split = int(len(X_hr) * 0.7)
        X_hr_tr, X_hr_te = X_hr[:split], X_hr[split:]
        y_real_tr, y_real_te = y_real[:split], y_real[split:]
        y_sim_tr, y_sim_te = y_sim[:split], y_sim[split:]
        tsc_tr, tsc_te = tsc[:split], tsc[split:]
        cv_tr, cv_te = calib_val[:split], calib_val[split:]

        # Normalization
        hr_mean, hr_std = X_hr_tr.mean(), X_hr_tr.std() + 1e-6
        X_hr_tr_n, X_hr_te_n = (X_hr_tr - hr_mean) / hr_std, (X_hr_te - hr_mean) / hr_std

        tsc_mean, tsc_std = tsc_tr.mean(), tsc_tr.std() + 1e-6
        tsc_tr_n, tsc_te_n = (tsc_tr - tsc_mean) / tsc_std, (tsc_te - tsc_mean) / tsc_std

        cv_mean, cv_std = cv_tr.mean(), cv_tr.std() + 1e-6
        cv_tr_n, cv_te_n = (cv_tr - cv_mean) / cv_std, (cv_te - cv_mean) / cv_std

        y_tr_res = y_real_tr - y_sim_tr

        loader = DataLoader(
            SelectiveWarmStartDataset(X_hr_tr_n, tsc_tr_n, cv_tr_n, y_tr_res),
            batch_size=32, shuffle=True
        )

        model = SelectiveWarmStartResidualLSTM(hidden_dim=16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
        loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(1, 101):
            for xb_hr, tscb, cvb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb_hr, tscb, cvb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            X_hr_te_t = torch.tensor(X_hr_te_n, dtype=torch.float32)
            tsc_te_t = torch.tensor(tsc_te_n, dtype=torch.float32)
            cv_te_t = torch.tensor(cv_te_n, dtype=torch.float32)
            pred_res = model(X_hr_te_t, tsc_te_t, cv_te_t).numpy()

        pred_final = y_sim_te + pred_res

        rmse_sim = np.sqrt(np.mean((y_real_te - y_sim_te) ** 2))
        rmse_hybrid = np.sqrt(np.mean((y_real_te - pred_final) ** 2))
        mae_sim = np.mean(np.abs(y_real_te - y_sim_te))
        mae_hybrid = np.mean(np.abs(y_real_te - pred_final))
        pct_reduction = ((rmse_sim - rmse_hybrid) / rmse_sim) * 100

        results[mode] = {
            "gb_val": 138.56 if custom_gb is None else round(custom_gb, 1),
            "mech_rmse": round(rmse_sim, 2),
            "mech_mae": round(mae_sim, 2),
            "hybrid_rmse": round(rmse_hybrid, 2),
            "hybrid_mae": round(mae_hybrid, 2),
            "pct_reduction": round(pct_reduction, 2)
        }

    print(f"\n--- STANDARDIZED RESULTS FOR PATIENT {patient_id} ---")
    print(f"Default Gb    (138.6 mg/dL): Physics RMSE = {results['default_gb']['mech_rmse']} mg/dL | Hybrid RMSE = {results['default_gb']['hybrid_rmse']} mg/dL (Gain: {results['default_gb']['pct_reduction']}%)")
    print(f"Personalized Gb ({results['personalized_gb']['gb_val']} mg/dL): Physics RMSE = {results['personalized_gb']['mech_rmse']} mg/dL | Hybrid RMSE = {results['personalized_gb']['hybrid_rmse']} mg/dL (Gain: {results['personalized_gb']['pct_reduction']}%)")
    print("=" * 75)


if __name__ == "__main__":
    evaluate_patient_gb("1031", "results/patient_1031_real_cgm_with_hr.csv")
    evaluate_patient_gb("1205", "results/patient_1205_real_cgm_hr.csv")