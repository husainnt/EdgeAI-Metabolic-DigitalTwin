"""
Late-Fusion Separate Stream Encoders + Modal Gating for Patient 1031.
1. Runs independent LSTM stream encoders for HR and Steps to prevent hidden state corruption.
2. Applies a learned modal gating mechanism to dynamically down-weight steps during sedentary periods.
3. Evaluates strictly against the 13-state simglucose ODE baseline at the +30-minute forecast horizon.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import verified simglucose ODE warm-start utilities
from run_selective_warmstart_simglucose_hybrid import (
    build_calibrated_t2d_patient,
    warm_start,
    simulate_open_loop,
    T2DPancreaticController
)

DATASET_PATH = "results/patient_1031_real_cgm_hr_steps.csv"
WINDOW_SIZE = 12  # 1-hour lookback (12 * 5min)
PRED_HORIZON = 6  # 30-min forecast horizon

# Set deterministic random seeds
torch.manual_seed(42)
np.random.seed(42)


class GatedSeparateEncoderResidualLSTM(nn.Module):
    def __init__(self, context_dim=2, hidden_dim=32, pred_horizon=6):
        super().__init__()
        # Independent stream encoders for Heart Rate and Steps
        self.lstm_hr = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.lstm_steps = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        
        # Dense encoder for sparse calibration context
        self.fc_context = nn.Linear(context_dim, 16)
        
        # Learned gating layer to dynamically scale modality weights
        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 16, 2),
            nn.Sigmoid()
        )
        
        # Final fusion layer predicting the 30-min horizon
        self.fc_out = nn.Linear(hidden_dim * 2 + 16, pred_horizon)
        self.relu = nn.ReLU()

    def forward(self, hr_seq, steps_seq, context_x):
        # Extract separate embeddings from HR and Steps LSTMs
        _, (h_hr, _) = self.lstm_hr(hr_seq)
        _, (h_steps, _) = self.lstm_steps(steps_seq)
        
        feat_hr = h_hr[-1]         # [batch, 32]
        feat_steps = h_steps[-1]   # [batch, 32]
        feat_ctx = self.relu(self.fc_context(context_x))  # [batch, 16]
        
        # Calculate dynamic modal weights
        combined_raw = torch.cat((feat_hr, feat_steps, feat_ctx), dim=1)
        gates = self.gate_layer(combined_raw)  # [batch, 2] -> (weight_hr, weight_steps)
        
        # Weight each feature stream independently before late fusion
        gated_hr = feat_hr * gates[:, 0:1]
        gated_steps = feat_steps * gates[:, 1:2]
        
        # Combine uncorrupted gated embeddings for final prediction
        fused = torch.cat((gated_hr, gated_steps, feat_ctx), dim=1)
        return self.fc_out(fused)


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

    calib_indices = sorted(list(set(calib_indices)))
    return calib_indices


def generate_real_simglucose_baseline_and_dataset(df, calib_indices):
    glucose = df['glucose_mg_dl'].values
    hr = df['heart_rate'].values
    steps = df['steps'].values
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

    X_hr, X_steps, X_ctx, y_res, y_true, y_sim = [], [], [], [], [], []

    for i in range(WINDOW_SIZE, num_samples - PRED_HORIZON):
        if simglucose_baseline[i] == 0 or simglucose_baseline[i + PRED_HORIZON - 1] == 0:
            continue

        hr_seq = hr[i - WINDOW_SIZE : i].reshape(-1, 1)
        steps_seq = steps[i - WINDOW_SIZE : i].reshape(-1, 1)

        target_true = glucose[i : i + PRED_HORIZON]
        sim_pred = simglucose_baseline[i : i + PRED_HORIZON]
        residual = target_true - sim_pred

        past_calibs = [c for c in calib_indices if c <= (i - WINDOW_SIZE)]
        if not past_calibs:
            continue
        last_c = past_calibs[-1]
        tsc = (i - last_c) * 5
        calib_val = glucose[last_c]

        X_hr.append(hr_seq)
        X_steps.append(steps_seq)
        X_ctx.append([tsc, calib_val])
        y_res.append(residual)
        y_true.append(target_true)
        y_sim.append(sim_pred)

    return (np.array(X_hr), np.array(X_steps), np.array(X_ctx), 
            np.array(y_res), np.array(y_true), np.array(y_sim))


def run_evaluation():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset missing at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    print(f"[+] Loaded multimodal dataset with {len(df)} rows.")

    calib_indices = extract_2x_daily_calibrations(df)
    X_hr, X_steps, X_ctx, y_res, y_true, y_sim = generate_real_simglucose_baseline_and_dataset(df, calib_indices)

    split = int(len(X_hr) * 0.7)

    X_hr_train, X_hr_test = X_hr[:split], X_hr[split:]
    X_steps_train, X_steps_test = X_steps[:split], X_steps[split:]
    X_ctx_train, X_ctx_test = X_ctx[:split], X_ctx[split:]
    y_res_train, y_res_test = y_res[:split], y_res[split:]
    y_true_test, y_sim_test = y_true[split:], y_sim[split:]

    scaler_hr = StandardScaler()
    X_hr_train_scaled = scaler_hr.fit_transform(X_hr_train.reshape(-1, 1)).reshape(X_hr_train.shape)
    X_hr_test_scaled = scaler_hr.transform(X_hr_test.reshape(-1, 1)).reshape(X_hr_test.shape)

    scaler_steps = StandardScaler()
    X_steps_train_scaled = scaler_steps.fit_transform(X_steps_train.reshape(-1, 1)).reshape(X_steps_train.shape)
    X_steps_test_scaled = scaler_steps.transform(X_steps_test.reshape(-1, 1)).reshape(X_steps_test.shape)

    scaler_ctx = StandardScaler()
    X_ctx_train_scaled = scaler_ctx.fit_transform(X_ctx_train)
    X_ctx_test_scaled = scaler_ctx.transform(X_ctx_test)

    t_hr_train = torch.tensor(X_hr_train_scaled, dtype=torch.float32)
    t_steps_train = torch.tensor(X_steps_train_scaled, dtype=torch.float32)
    t_ctx_train = torch.tensor(X_ctx_train_scaled, dtype=torch.float32)
    t_res_train = torch.tensor(y_res_train, dtype=torch.float32)

    t_hr_test = torch.tensor(X_hr_test_scaled, dtype=torch.float32)
    t_steps_test = torch.tensor(X_steps_test_scaled, dtype=torch.float32)
    t_ctx_test = torch.tensor(X_ctx_test_scaled, dtype=torch.float32)

    model = GatedSeparateEncoderResidualLSTM(context_dim=2, hidden_dim=32, pred_horizon=PRED_HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        preds = model(t_hr_train, t_steps_train, t_ctx_train)
        loss = criterion(preds, t_res_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predicted_residuals = model(t_hr_test, t_steps_test, t_ctx_test).numpy()

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
    print("   LATE-FUSION GATED DUAL-STREAM EVALUATION (+30 MIN HORIZON)   ")
    print("="*60)
    print(f"Mechanistic Open-Loop Baseline RMSE: {mech_rmse:.2f} mg/dL  (MAE: {mech_mae:.2f})")
    print(f"Gated Dual-Stream Hybrid RMSE:      {hybrid_rmse:.2f} mg/dL  (MAE: {hybrid_mae:.2f})")
    print(f"Error Reduction over Physics:      {pct_reduction:.2f}%")
    print("="*65)


if __name__ == "__main__":
    run_evaluation()