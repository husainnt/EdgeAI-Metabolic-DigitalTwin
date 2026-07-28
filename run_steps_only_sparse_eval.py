"""
Corrected Steps-Only Sparse Multimodal Evaluation for Patient 1031.
Reuses the exact open-loop simglucose ODE engine from `run_selective_warmstart_simglucose_hybrid.py`.
Evaluates specifically at the +30-minute prediction horizon (index -1) for direct parity with HR benchmark.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Import real working functions and controller from your tested HR script
from run_selective_warmstart_simglucose_hybrid import (
    build_calibrated_t2d_patient,
    warm_start,
    simulate_open_loop,
    T2DPancreaticController
)

DATASET_PATH = "results/patient_1031_real_cgm_hr_steps.csv"
WINDOW_SIZE = 12  # 1-hour lookback (12 * 5min)
PRED_HORIZON = 6  # 30-min forecast horizon

torch.manual_seed(42)
np.random.seed(42)


class StepsSparseResidualLSTM(nn.Module):
    """
    Residual LSTM mapping sparse context (tsc, calib_val) + continuous Steps sequence 
    to the open-loop simglucose error gap.
    """
    def __init__(self, seq_dim=1, context_dim=2, hidden_dim=32, pred_horizon=6):
        super(StepsSparseResidualLSTM, self).__init__()
        self.lstm = nn.LSTM(seq_dim, hidden_dim, batch_first=True)
        self.fc_context = nn.Linear(context_dim, 16)
        self.fc_out = nn.Linear(hidden_dim + 16, pred_horizon)
        self.relu = nn.ReLU()

    def forward(self, seq_x, context_x):
        _, (h_n, _) = self.lstm(seq_x)
        lstm_feat = h_n[-1]  # [batch, hidden_dim]
        
        ctx_feat = self.relu(self.fc_context(context_x))  # [batch, 16]
        combined = torch.cat((lstm_feat, ctx_feat), dim=1)
        
        out = self.fc_out(combined)
        return out


def extract_2x_daily_calibrations(df, window_tolerance_min=45):
    """
    Identifies exact 2x/day calibration indices (~8 AM and ~8 PM) 
    using single-nearest-reading matching (matching the real HR script).
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    dates = df['timestamp'].dt.date.unique()
    calib_indices = []

    for d in dates:
        day_df = df[df['timestamp'].dt.date == d]
        
        # Target 1: 08:00 AM
        t8 = pd.to_datetime(f"{d} 08:00:00")
        diffs8 = (day_df['timestamp'] - t8).abs()
        if not diffs8.empty and diffs8.min() <= pd.Timedelta(minutes=window_tolerance_min):
            calib_indices.append(diffs8.idxmin())
            
        # Target 2: 08:00 PM (20:00)
        t20 = pd.to_datetime(f"{d} 20:00:00")
        diffs20 = (day_df['timestamp'] - t20).abs()
        if not diffs20.empty and diffs20.min() <= pd.Timedelta(minutes=window_tolerance_min):
            calib_indices.append(diffs20.idxmin())

    calib_indices = sorted(list(set(calib_indices)))
    print(f"[+] Found {len(calib_indices)} sparse 2x/day calibration points across dataset.")
    return calib_indices


def generate_real_simglucose_baseline_and_dataset(df, calib_indices):
    """
    Runs actual simglucose ODE warm-start across sparse segments using exact signatures.
    """
    glucose = df['glucose_mg_dl'].values
    steps = df['steps'].values
    num_samples = len(df)

    simglucose_baseline = np.zeros(num_samples)

    # Initialize patient and T2D controller matching working HR setup
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

    # Build dataset features (STEPS sequence ONLY, NO continuous CGM leakage)
    X_seq, X_ctx, y_res, y_true, y_sim = [], [], [], [], []

    for i in range(WINDOW_SIZE, num_samples - PRED_HORIZON):
        if simglucose_baseline[i] == 0 or simglucose_baseline[i + PRED_HORIZON - 1] == 0:
            continue

        steps_seq = steps[i - WINDOW_SIZE : i].reshape(-1, 1)
        target_true = glucose[i : i + PRED_HORIZON]
        sim_pred = simglucose_baseline[i : i + PRED_HORIZON]
        residual = target_true - sim_pred

        past_calibs = [c for c in calib_indices if c <= (i - WINDOW_SIZE)]
        if not past_calibs:
            continue
        last_c = past_calibs[-1]
        tsc = (i - last_c) * 5  # minutes
        calib_val = glucose[last_c]

        X_seq.append(steps_seq)
        X_ctx.append([tsc, calib_val])
        y_res.append(residual)
        y_true.append(target_true)
        y_sim.append(sim_pred)

    return (np.array(X_seq), np.array(X_ctx), 
            np.array(y_res), np.array(y_true), np.array(y_sim))


def run_evaluation():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset missing at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    print(f"[+] Loaded multimodal dataset with {len(df)} rows.")

    calib_indices = extract_2x_daily_calibrations(df)
    X_seq, X_ctx, y_res, y_true, y_sim = generate_real_simglucose_baseline_and_dataset(df, calib_indices)

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

    model = StepsSparseResidualLSTM(seq_dim=1, context_dim=2, hidden_dim=32, pred_horizon=PRED_HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        preds = model(t_seq_train, t_ctx_train)
        loss = criterion(preds, t_res_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predicted_residuals = model(t_seq_test, t_ctx_test).numpy()

    hybrid_preds = y_sim_test + predicted_residuals

    # Evaluate strictly at the +30 minute horizon (index -1)
    y_true_30min = y_true_test[:, -1]
    y_sim_30min = y_sim_test[:, -1]
    hybrid_30min = hybrid_preds[:, -1]

    mech_rmse = np.sqrt(np.mean((y_true_30min - y_sim_30min) ** 2))
    hybrid_rmse = np.sqrt(np.mean((y_true_30min - hybrid_30min) ** 2))
    mech_mae = np.mean(np.abs(y_true_30min - y_sim_30min))
    hybrid_mae = np.mean(np.abs(y_true_30min - hybrid_30min))
    pct_reduction = ((mech_rmse - hybrid_rmse) / mech_rmse) * 100

    print("\n" + "="*60)
    print("   REAL SIMGLUCOSE SPARSE STEPS-ONLY EVALUATION (+30 MIN HORIZON)   ")
    print("="*60)
    print(f"Mechanistic Open-Loop Baseline RMSE: {mech_rmse:.2f} mg/dL  (MAE: {mech_mae:.2f})")
    print(f"Steps-Only Hybrid RMSE:            {hybrid_rmse:.2f} mg/dL  (MAE: {hybrid_mae:.2f})")
    print(f"Error Reduction:                   {pct_reduction:.2f}%")
    print("="*60)


if __name__ == "__main__":
    run_evaluation()