"""
Evaluation script to benchmark Short-Term HR Volatility (Autonomic Tone Proxy)
against the 13-state simglucose ODE baseline for Patient 1031 (+30 min horizon).
Matches 100 Epochs + CosineAnnealingLR setup.
"""

import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from run_selective_warmstart_simglucose_hybrid import (
    build_calibrated_t2d_patient,
    warm_start,
    simulate_open_loop,
    T2DPancreaticController
)

DATASET_PATH = "results/patient_1031_real_cgm_hr_steps_hrv.csv"
WINDOW_SIZE = 12  # 1-hour lookback (12 * 5min)
PRED_HORIZON = 6  # 30-min forecast horizon

torch.manual_seed(42)
np.random.seed(42)


class StandaloneResidualLSTM(nn.Module):
    def __init__(self, feature_dim=1, context_dim=2, hidden_dim=32, pred_horizon=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_context = nn.Linear(context_dim, 16)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim + 16, 32),
            nn.ReLU(),
            nn.Linear(32, pred_horizon)
        )

    def forward(self, x_seq, context_x):
        _, (h_n, _) = self.lstm(x_seq)
        h_last = h_n[-1]
        ctx_emb = torch.relu(self.fc_context(context_x))
        combined = torch.cat((h_last, ctx_emb), dim=1)
        return self.fc_out(combined)


def extract_2x_daily_calibrations(df, window_tolerance_min=45):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    dates = df['timestamp'].dt.date.unique()
    calib_indices = []

    for d in dates:
        day_df = df[df['timestamp'].dt.date == d]
        t8 = pd.to_datetime(f"{d} 08:00:00")
        diffs8 = (day_df['timestamp'] - t8).abs()
        if not diffs8.empty and diffs8.min() <= pd.Timedelta(minutes=window_tolerance_min):
            calib_indices.append(diffs8.idxmin())
            
        t20 = pd.to_datetime(f"{d} 20:00:00")
        diffs20 = (day_df['timestamp'] - t20).abs()
        if not diffs20.empty and diffs20.min() <= pd.Timedelta(minutes=window_tolerance_min):
            calib_indices.append(diffs20.idxmin())

    return sorted(list(set(calib_indices)))


def generate_dataset(df, calib_indices, feature_col='hr_volatility_30m'):
    glucose = df['glucose_mg_dl'].values
    feature_vals = df[feature_col].values
    num_samples = len(df)

    simglucose_baseline = np.zeros(num_samples)
    patient = build_calibrated_t2d_patient()
    params = patient._params
    basal_rate_correct = params['u2ss'] * params['BW'] / 6000
    controller = T2DPancreaticController(gb=params['Gb'], basal_rate=basal_rate_correct)

    segments = list(zip(calib_indices[:-1], calib_indices[1:]))
    if calib_indices[-1] < num_samples:
        segments.append((calib_indices[-1], num_samples))

    for idx_start, idx_end in segments:
        calib_val = glucose[idx_start]
        segment_len = idx_end - idx_start
        patient.reset()
        warm_start(patient, calib_val)
        sim_trace = simulate_open_loop(patient, controller, segment_len)
        simglucose_baseline[idx_start:idx_end] = sim_trace[:segment_len]

    X_seq, X_ctx, y_res, y_true, y_sim = [], [], [], [], []

    for i in range(WINDOW_SIZE, num_samples - PRED_HORIZON):
        if simglucose_baseline[i] == 0 or simglucose_baseline[i + PRED_HORIZON - 1] == 0:
            continue

        seq = feature_vals[i - WINDOW_SIZE : i].reshape(-1, 1)
        target_true = glucose[i : i + PRED_HORIZON]
        sim_pred = simglucose_baseline[i : i + PRED_HORIZON]
        residual = target_true - sim_pred

        past_calibs = [c for c in calib_indices if c <= (i - WINDOW_SIZE)]
        if not past_calibs:
            continue
        last_c = past_calibs[-1]
        tsc = (i - last_c) * 5
        calib_val = glucose[last_c]

        X_seq.append(seq)
        X_ctx.append([tsc, calib_val])
        y_res.append(residual)
        y_true.append(target_true)
        y_sim.append(sim_pred)

    return (np.array(X_seq), np.array(X_ctx), np.array(y_res), 
            np.array(y_true), np.array(y_sim))


def run_volatility_benchmark():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Run extract_and_merge_garmin_hrv.py first to create {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    print(f"[+] Loaded dataset with {len(df)} rows.")

    calib_indices = extract_2x_daily_calibrations(df)
    X_seq, X_ctx, y_res, y_true, y_sim = generate_dataset(df, calib_indices, feature_col='hr_volatility_30m')

    split = int(len(X_seq) * 0.7)
    X_seq_train, X_seq_test = X_seq[:split], X_seq[split:]
    X_ctx_train, X_ctx_test = X_ctx[:split], X_ctx[split:]
    y_res_train, y_res_test = y_res[:split], y_res[split:]
    y_true_test, y_sim_test = y_true[split:], y_sim[split:]

    scaler_seq = StandardScaler()
    X_seq_train_scaled = scaler_seq.fit_transform(X_seq_train.reshape(-1, 1)).reshape(X_seq_train.shape)
    X_seq_test_scaled = scaler_seq.transform(X_seq_test.reshape(-1, 1)).reshape(X_seq_test.shape)

    scaler_ctx = StandardScaler()
    X_ctx_train_scaled = scaler_ctx.fit_transform(X_ctx_train)
    X_ctx_test_scaled = scaler_ctx.transform(X_ctx_test)

    t_seq_train = torch.tensor(X_seq_train_scaled, dtype=torch.float32)
    t_ctx_train = torch.tensor(X_ctx_train_scaled, dtype=torch.float32)
    t_res_train = torch.tensor(y_res_train, dtype=torch.float32)

    t_seq_test = torch.tensor(X_seq_test_scaled, dtype=torch.float32)
    t_ctx_test = torch.tensor(X_ctx_test_scaled, dtype=torch.float32)

    model = StandaloneResidualLSTM(feature_dim=1, context_dim=2, hidden_dim=32, pred_horizon=PRED_HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    epochs = 100
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    model.train()
    print("\n" + "="*65)
    print(" Training Standalone HR Volatility Model (100 Epochs)...")
    print("="*65)
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        preds = model(t_seq_train, t_ctx_train)
        loss = criterion(preds, t_res_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:03d}/{epochs} | Train MSE Loss: {loss.item():.2f} | LR: {current_lr:.6f}")

    model.eval()
    with torch.no_grad():
        predicted_residuals = model(t_seq_test, t_ctx_test).numpy()

    hybrid_preds = y_sim_test + predicted_residuals
    y_true_30min = y_true_test[:, -1]
    y_sim_30min = y_sim_test[:, -1]
    hybrid_30min = hybrid_preds[:, -1]

    mech_rmse = np.sqrt(np.mean((y_true_30min - y_sim_30min) ** 2))
    hybrid_rmse = np.sqrt(np.mean((y_true_30min - hybrid_30min) ** 2))
    mech_mae = np.mean(np.abs(y_true_30min - y_sim_30min))
    hybrid_mae = np.mean(np.abs(y_true_30min - hybrid_30min))
    pct_reduction = ((mech_rmse - hybrid_rmse) / mech_rmse) * 100

    print("\n" + "="*65)
    print("   STANDALONE SHORT-TERM HR VOLATILITY EVALUATION (+30 MIN)")
    print("="*65)
    print(f"Mechanistic Open-Loop Baseline RMSE: {mech_rmse:.2f} mg/dL  (MAE: {mech_mae:.2f})")
    print(f"HR Volatility Hybrid RMSE:           {hybrid_rmse:.2f} mg/dL  (MAE: {hybrid_mae:.2f})")
    print(f"Error Reduction over Physics:        {pct_reduction:.2f}%")
    print("="*65)


if __name__ == "__main__":
    run_volatility_benchmark()