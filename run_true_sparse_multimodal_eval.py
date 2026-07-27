"""
True CGM-Free Multimodal Evaluation — Patient 1031
--------------------------------------------------
Strictly enforces the sparse-calibration deployment scenario:
- NO continuous CGM history is fed into the model.
- Continuous Sequence Input: 60 minutes of real Garmin HR telemetry ONLY.
- Calibration Context Input: Last known fingerstick value (G_calib) + time elapsed (dt).
- Baseline: Forward-filled fingerstick value (simulating open-loop drift).
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

CSV_PATH = "results/patient_1031_real_cgm_with_hr.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Could not find {CSV_PATH}. Run merge script first.")

df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
df = df.sort_values("timestamp").reset_index(drop=True)

actual_days = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 86400
print(f"Loaded {len(df)} readings for Patient 1031 (Span: {actual_days:.1f} days)")

cgm_cols = [c for c in df.columns if any(k in c.lower() for k in ['cgm', 'glucose', 'val', 'reading', 'mg/dl'])]
hr_cols = [c for c in df.columns if any(k in c.lower() for k in ['hr', 'heart', 'bpm', 'rate'])]

glucose = df[cgm_cols[0]].values.astype(np.float32)
hr_data = df[hr_cols[0]].values.astype(np.float32)
df["date_str"] = df["timestamp"].dt.strftime("%Y-%m-%d")

LOOKBACK = 12  # 60 mins of Garmin HR history
HORIZON = 6   # 30 mins ahead forecast

# Extract 2x/day sparse calibration points (~8:00 AM & 8:00 PM)
calibration_indices = set()
for date_str, group in df.groupby("date_str"):
    for target_hour in [8, 20]:
        target_time = pd.to_datetime(f"{date_str} {target_hour:02d}:00:00")
        diffs = (group["timestamp"] - target_time).abs()
        closest_idx = diffs.idxmin()
        if diffs.loc[closest_idx] <= pd.Timedelta(minutes=45):
            calibration_indices.add(closest_idx)

calibration_indices = sorted(list(calibration_indices))
print(f"Extracted {len(calibration_indices)} real sparse fingerstick calibration points (~8am/8pm)")

# Compute sparse open-loop state (Forward-filled fingerstick + elapsed time)
time_since_calib_series = np.zeros(len(glucose), dtype=np.float32)
calib_val_series = np.zeros(len(glucose), dtype=np.float32)
last_val, tsc = glucose[0], 0.0

for i in range(len(glucose)):
    if i in calibration_indices:
        last_val = glucose[i]
        tsc = 0.0
    else:
        tsc += 5.0 / 60.0  # 5-min steps -> hours
    time_since_calib_series[i] = tsc
    calib_val_series[i] = last_val

# Construct Dataset Windows WITHOUT Continuous Glucose Sequence
X_hr_seq, y_real, y_base, tsc_arr, calib_arr = [], [], [], [], []

for i in range(LOOKBACK, len(glucose) - HORIZON):
    # Only HR is a continuous sequence input!
    X_hr_seq.append(hr_data[i - LOOKBACK:i])
    
    # Target is 30 mins ahead
    y_real.append(glucose[i + HORIZON])
    
    # Baseline is forward-filled fingerstick (true open-loop estimate)
    y_base.append(calib_val_series[i + HORIZON])
    
    tsc_arr.append(time_since_calib_series[i + HORIZON])
    calib_arr.append(calib_val_series[i + HORIZON])

X_hr_seq = np.array(X_hr_seq, dtype=np.float32)
y_real = np.array(y_real, dtype=np.float32)
y_base = np.array(y_base, dtype=np.float32)
tsc_arr = np.array(tsc_arr, dtype=np.float32)
calib_arr = np.array(calib_arr, dtype=np.float32)

# Train/Test Split (70% Train / 30% Test chronologically)
split_idx = int(len(X_hr_seq) * 0.7)

X_hr_train, X_hr_test = X_hr_seq[:split_idx], X_hr_seq[split_idx:]
y_real_train, y_real_test = y_real[:split_idx], y_real[split_idx:]
y_base_train, y_base_test = y_base[:split_idx], y_base[split_idx:]
tsc_train, tsc_test = tsc_arr[:split_idx], tsc_arr[split_idx:]
calibval_train, calibval_test = calib_arr[:split_idx], calib_arr[split_idx:]

# Feature Normalization
hr_mean, hr_std = X_hr_train.mean(), X_hr_train.std() + 1e-6
X_hr_tr_n = (X_hr_train - hr_mean) / hr_std
X_hr_te_n = (X_hr_test - hr_mean) / hr_std

tsc_mean, tsc_std = tsc_train.mean(), tsc_train.std() + 1e-6
tsc_tr_n, tsc_te_n = (tsc_train - tsc_mean) / tsc_std, (tsc_test - tsc_mean) / tsc_std

cv_mean, cv_std = calibval_train.mean(), calibval_train.std() + 1e-6
cv_tr_n, cv_te_n = (calibval_train - cv_mean) / cv_std, (calibval_test - cv_mean) / cv_std

# Target for ML head: Residual error between real glucose and forward-filled fingerstick
y_train_residual = y_real_train - y_base_train

class TrueSparseMultimodalDataset(Dataset):
    def __init__(self, X_hr, tsc, calibval, y):
        self.X_hr = torch.tensor(X_hr, dtype=torch.float32)
        self.tsc = torch.tensor(tsc, dtype=torch.float32)
        self.calibval = torch.tensor(calibval, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X_hr)
    def __getitem__(self, idx): return self.X_hr[idx], self.tsc[idx], self.calibval[idx], self.y[idx]

class TrueSparseMultimodalLSTM(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        # Encodes 60-min continuous Heart Rate sequence
        self.hr_encoder = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        
        # Fuses HR embedding + Last Fingerstick Value + Elapsed Time
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x_hr, tsc, calibval):
        x_hr = x_hr.unsqueeze(-1)  # shape (B, 12, 1)
        _, (h_n, _) = self.hr_encoder(x_hr)
        h_hr = h_n.squeeze(0)      # shape (B, hidden_dim)
        
        # Combine HR feature with sparse calibration context
        combined = torch.cat([h_hr, tsc.unsqueeze(-1), calibval.unsqueeze(-1)], dim=1)
        return self.fusion_head(combined).squeeze(-1)

loader = DataLoader(
    TrueSparseMultimodalDataset(X_hr_tr_n, tsc_tr_n, cv_tr_n, y_train_residual), 
    batch_size=32, shuffle=True
)

model = TrueSparseMultimodalLSTM(hidden_dim=16)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn = nn.MSELoss()

print("\nTraining True CGM-Free Multimodal Model (Garmin HR + Sparse Fingerstick Context)...")
print("=" * 85)

model.train()
EPOCHS = 40
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    for xb_hr, tscb, cvb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb_hr, tscb, cvb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train MSE Loss: {epoch_loss/len(loader):.2f}")

# Evaluation on Held-Out Test Set
model.eval()
with torch.no_grad():
    X_hr_test_t = torch.tensor(X_hr_te_n, dtype=torch.float32)
    tsc_test_t = torch.tensor(tsc_te_n, dtype=torch.float32)
    cv_test_t = torch.tensor(cv_te_n, dtype=torch.float32)
    pred_res = model(X_hr_test_t, tsc_test_t, cv_test_t).numpy()

pred_final = y_base_test + pred_res

rmse_base = np.sqrt(np.mean((y_real_test - y_base_test) ** 2))
rmse_hyb = np.sqrt(np.mean((y_real_test - pred_final) ** 2))
mae_base = np.mean(np.abs(y_real_test - y_base_test))
mae_hyb = np.mean(np.abs(y_real_test - pred_final))
improvement_pct = ((rmse_base - rmse_hyb) / rmse_base) * 100

print("=" * 85)
print(f"HONEST CGM-FREE RESULTS: 2x/day Fingerstick + Continuous Garmin HR ({actual_days:.1f} Days)")
print("=" * 85)
print(f"  • Open-Loop Baseline RMSE (Forward-filled Fingerstick): {rmse_base:.2f} mg/dL | MAE: {mae_base:.2f} mg/dL")
print(f"  • Hybrid Model RMSE (Fingerstick + Real Garmin HR):      {rmse_hyb:.2f} mg/dL | MAE: {mae_hyb:.2f} mg/dL")
print(f"  • Honest RMSE Improvement:                               {improvement_pct:+.1f}%")
print("=" * 85)

# Save Diagnostic Plot
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(14, 6))
plot_n = min(300, len(y_real_test))
plt.plot(y_real_test[:plot_n], label="Real Glucose (Ground Truth)", color="black", linewidth=1.5)
plt.plot(y_base_test[:plot_n], label="Open-Loop Baseline (Fingerstick alone)", color="gray", linestyle="--", alpha=0.7)
plt.plot(pred_final[:plot_n], label="Hybrid Model (Fingerstick + Garmin HR)", color="green", alpha=0.85)
plt.xlabel("Test Sample Index (Chronological)")
plt.ylabel("Glucose (mg/dL)")
plt.title(f"True CGM-Free Evaluation — Patient 1031 (First {plot_n} Test Points)")
plt.legend()
plt.tight_layout()
plt.savefig("results/true_sparse_multimodal_eval.png", dpi=120)
print("[SAVED] Diagnostic plot: results/true_sparse_multimodal_eval.png")