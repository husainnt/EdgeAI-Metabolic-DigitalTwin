"""
Standalone Steps-Only Evaluation for Patient 1031.
Evaluates forecasting performance using CGM + Steps (excluding Heart Rate)
to establish an isolated single-modality benchmark against the mechanistic baseline.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "results/patient_1031_real_cgm_hr_steps.csv"
WINDOW_SIZE = 12  # 1 hour look-back (12 * 5min)
PRED_HORIZON = 6  # 30-min prediction horizon

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class StepsResidualLSTM(nn.Module):
    """
    LSTM residual network mapping [CGM, Steps] windows to future glucose residual error.
    """
    def __init__(self, input_dim=2, hidden_dim=32, num_layers=1):
        super(StepsResidualLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, PRED_HORIZON)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def create_sequences(glucose, steps, window_size, pred_horizon):
    X, y_base_err, y_true = [], [], []
    
    # Simple mechanistic baseline proxy (persistence/drift assumption over sparse windows)
    for i in range(len(glucose) - window_size - pred_horizon):
        cgm_seq = glucose[i : i + window_size]
        steps_seq = steps[i : i + window_size]
        
        # Ground truth future glucose
        target_true = glucose[i + window_size : i + window_size + pred_horizon]
        
        # Mechanistic baseline prediction (last known CGM value carried forward)
        mech_pred = np.full(pred_horizon, cgm_seq[-1])
        
        # Residual target for LSTM to learn: Ground_Truth - Mech_Baseline
        residual_target = target_true - mech_pred
        
        feature_matrix = np.column_stack((cgm_seq, steps_seq))
        X.append(feature_matrix)
        y_base_err.append(residual_target)
        y_true.append(target_true)
        
    return np.array(X), np.array(y_base_err), np.array(y_true)


def run_evaluation():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    glucose = df['glucose_mg_dl'].values
    steps = df['steps'].values

    print(f"[+] Loaded dataset with {len(df)} rows.")
    print(f"Active step windows (>0 steps): {(steps > 0).sum()} / {len(steps)} ({((steps > 0).sum()/len(steps))*100:.1f}%)")

    # Create sliding window sequences
    X, y_residual, y_true = create_sequences(glucose, steps, WINDOW_SIZE, PRED_HORIZON)

    # Train / Test split (80 / 20 chronologically)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_res_train, y_res_test = y_residual[:split_idx], y_residual[split_idx:]
    y_true_test = y_true[split_idx:]

    # Normalize input features
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_flat)

    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Convert to Tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_res_train_t = torch.tensor(y_res_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

    # Initialize model, loss, optimizer
    model = StepsResidualLSTM(input_dim=2, hidden_dim=32)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # Train residual model
    model.train()
    epochs = 40
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = criterion(preds, y_res_train_t)
        loss.backward()
        optimizer.step()

    # Evaluate on Test Set
    model.eval()
    with torch.no_grad():
        predicted_residuals = model(X_test_t).numpy()

    # Reconstruct predictions: Mech_Baseline + Model_Residual_Correction
    mech_baseline_test = y_true_test - y_res_test
    hybrid_steps_preds = mech_baseline_test + predicted_residuals

    # Calculate RMSE Metrics
    mech_rmse = np.sqrt(np.mean((y_true_test - mech_baseline_test) ** 2))
    hybrid_steps_rmse = np.sqrt(np.mean((y_true_test - hybrid_steps_preds) ** 2))
    pct_improvement = ((mech_rmse - hybrid_steps_rmse) / mech_rmse) * 100

    print("\n" + "="*50)
    print("      STANDALONE STEPS-ONLY EVALUATION RESULTS      ")
    print("="*50)
    print(f"Mechanistic Baseline RMSE:  {mech_rmse:.2f} mg/dL")
    print(f"Steps-Only Hybrid RMSE:     {hybrid_steps_rmse:.2f} mg/dL")
    print(f"Error Reduction:            {pct_improvement:.2f}%")
    print("="*50)


if __name__ == "__main__":
    run_evaluation()