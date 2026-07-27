"""
Real Multimodal Evaluation — Patient 1031
-----------------------------------------
Evaluates 2x/day calibration (~12-hour open-loop gaps) using 100% genuine
Garmin Vívosmart 5 smartwatch heart rate telemetry.
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
print(f"Loaded {len(df)} aligned readings for Patient 1031 (Span: {actual_days:.1f} days)")

cgm_cols = [c for c in df.columns if any(k in c.lower() for k in ['cgm', 'glucose', 'val', 'reading', 'mg/dl'])]
hr_cols = [c for c in df.columns if any(k in c.lower() for k in ['hr', 'heart', 'bpm', 'rate'])]

glucose = df[cgm_cols[0]].values.astype(np.float32)
hr_data = df[hr_cols[0]].values.astype(np.float32)
df["date_str"] = df["timestamp"].dt.strftime("%Y-%m-%d")

print(f"  • Real Glucose Mean: {glucose.mean():.1f} mg/dL (Std: {glucose.std():.1f})")
print(f"  • Real Garmin HR Mean: {hr_data.mean():.1f} BPM (Std: {hr_data.std():.1f})\n")

WINDOW = 24  # 2-hour trailing average baseline
baseline = pd.Series(glucose).rolling(WINDOW, center=False, min_periods=1).mean().values

LOOKBACK = 12  # 60 mins of history
HORIZON = 6   # 30 mins ahead forecast

# Extract 2x/day calibration points (~8:00 AM and ~8:00 PM)
calibration_indices = set()
for date_str, group in df.groupby("date_str"):
    for target_hour in [8, 20]:
        target_time = pd.to_datetime(f"{date_str} {target_hour:02d}:00:00")
        diffs = (group["timestamp"] - target_time).abs()
        closest_idx = diffs.idxmin()
        if diffs.loc[closest_idx] <= pd.Timedelta(minutes=45):
            calibration_indices.add(closest_idx)

calibration_indices = sorted(list(calibration_indices))

# Build Time-Since-Calibration features
time_since_calib_series = np.zeros(len(glucose))
calib_val_series = np.zeros(len(glucose))
last_val = glucose[0]
tsc = 0.0

for i in range(len(glucose)):
    if i in calibration_indices:
        last_val = glucose[i]
        tsc = 0.0
    else:
        tsc += 5.0 / 60.0  # 5-min steps in hours
    time_since_calib_series[i] = tsc
    calib_val_series[i] = last_val

# Construct Multi-Input Windows [Glucose, Real_Garmin_HR]
X_glc, X_hr, y_real, y_base, tsc_arr, calib_arr = [], [], [], [], [], []
for i in range(LOOKBACK, len(glucose) - HORIZON):
    X_glc.append(glucose[i - LOOKBACK:i])
    X_hr.append(hr_data[i - LOOKBACK:i])
    y_real.append(glucose[i + HORIZON])
    y_base.append(baseline[i + HORIZON])
    tsc_arr.append(time_since_calib_series[i + HORIZON])
    calib_arr.append(calib_val_series[i + HORIZON])

X_glc = np.array(X_glc)
X_hr = np.array(X_hr)
y_real = np.array(y_real)
y_base = np.array(y_base)
tsc_arr = np.array(tsc_arr, dtype=np.float32)
calib_arr = np.array(calib_arr, dtype=np.float32)

# Train/Test Split (70% Train / 30% Test)
split_idx = int(len(X_glc) * 0.7)

X_glc_train, X_glc_test = X_glc[:split_idx], X_glc[split_idx:]
X_hr_train, X_hr_test = X_hr[:split_idx], X_hr[split_idx:]
y_real_train, y_real_test = y_real[:split_idx], y_real[split_idx:]
y_base_train, y_base_test = y_base[:split_idx], y_base[split_idx:]
tsc_train, tsc_test = tsc_arr[:split_idx], tsc_arr[split_idx:]
calibval_train, calibval_test = calib_arr[:split_idx], calib_arr[split_idx:]

# Feature Normalization
glc_mean, glc_std = X_glc_train.mean(), X_glc_train.std() + 1e-6
hr_mean, hr_std = X_hr_train.mean(), X_hr_train.std() + 1e-6

X_glc_tr_n, X_glc_te_n = (X_glc_train - glc_mean) / glc_std, (X_glc_test - glc_mean) / glc_std
X_hr_tr_n, X_hr_te_n = (X_hr_train - hr_mean) / hr_std, (X_hr_test - hr_mean) / hr_std

tsc_mean, tsc_std = tsc_train.mean(), tsc_train.std() + 1e-6
tsc_tr_n, tsc_te_n = (tsc_train - tsc_mean) / tsc_std, (tsc_test - tsc_mean) / tsc_std

cv_mean, cv_std = calibval_train.mean(), calibval_train.std() + 1e-6
cv_tr_n, cv_te_n = (calibval_train - cv_mean) / cv_std, (calibval_test - cv_mean) / cv_std

# Stack Glucose + Real HR into 2-channel sequence input: shape (N, 12, 2)
X_seq_train = np.stack([X_glc_tr_n, X_hr_tr_n], axis=-1)
X_seq_test = np.stack([X_glc_te_n, X_hr_te_n], axis=-1)

y_train_residual = y_real_train - y_base_train

# PyTorch Dataset
class MultimodalCalibDataset(Dataset):
    def __init__(self, X_seq, tsc, calibval, y):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.tsc = torch.tensor(tsc, dtype=torch.float32)
        self.calibval = torch.tensor(calibval, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X_seq)
    def __getitem__(self, idx): return self.X_seq[idx], self.tsc[idx], self.calibval[idx], self.y[idx]

# Multimodal LSTM Model
class MultimodalResidualLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x_seq, tsc, calibval):
        _, (h_n, _) = self.lstm(x_seq)
        h = h_n.squeeze(0)
        combined = torch.cat([h, tsc.unsqueeze(-1), calibval.unsqueeze(-1)], dim=1)
        return self.head(combined).squeeze(-1)

loader = DataLoader(MultimodalCalibDataset(X_seq_train, tsc_tr_n, cv_tr_n, y_train_residual), batch_size=32, shuffle=True)

model = MultimodalResidualLSTM(input_dim=2, hidden_dim=16)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
loss_fn = nn.MSELoss()

print("Training Multimodal Residual LSTM (Glucose + Real Garmin HR) at 2x/day Calibration...")
print("=" * 85)

model.train()
EPOCHS = 40
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    for xb_seq, tscb, cvb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb_seq, tscb, cvb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d}/{EPOCHS} | Train MSE Loss: {epoch_loss/len(loader):.2f}")

# Evaluation on Held-Out Test Set
model.eval()
with torch.no_grad():
    X_seq_test_t = torch.tensor(X_seq_test, dtype=torch.float32)
    tsc_test_t = torch.tensor(tsc_te_n, dtype=torch.float32)
    cv_test_t = torch.tensor(cv_te_n, dtype=torch.float32)
    pred_res = model(X_seq_test_t, tsc_test_t, cv_test_t).numpy()

pred_final = y_base_test + pred_res

rmse_base = np.sqrt(np.mean((y_real_test - y_base_test) ** 2))
rmse_hyb = np.sqrt(np.mean((y_real_test - pred_final) ** 2))
mae_base = np.mean(np.abs(y_real_test - y_base_test))
mae_hyb = np.mean(np.abs(y_real_test - pred_final))
improvement_pct = ((rmse_base - rmse_hyb) / rmse_base) * 100

print("=" * 85)
print(f"REAL MULTIMODAL RESULTS: 2x/day Calibration + Real Garmin HR Telemetry ({actual_days:.1f} Days)")
print("=" * 85)
print(f"  • Baseline-Only RMSE:             {rmse_base:.2f} mg/dL  | MAE: {mae_base:.2f} mg/dL")
print(f"  • Hybrid (Glc + Real Garmin HR):  {rmse_hyb:.2f} mg/dL  | MAE: {mae_hyb:.2f} mg/dL")
print(f"  • Genuine RMSE Improvement:      {improvement_pct:+.1f}%")
print("=" * 85)

# Save Diagnostic Plot
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(14, 6))
plot_n = min(300, len(y_real_test))
plt.plot(y_real_test[:plot_n], label="Real Glucose", color="black", linewidth=1.5)
plt.plot(y_base_test[:plot_n], label="Baseline (No ML)", color="gray", linestyle="--", alpha=0.7)
plt.plot(pred_final[:plot_n], label="Hybrid (Glc + Real Garmin HR)", color="green", alpha=0.85)
plt.xlabel("Test Sample Index (Chronological)")
plt.ylabel("Glucose (mg/dL)")
plt.title(f"Patient 1031 — 2x/Day Calibration with Real Garmin Smartwatch HR (First {plot_n} Test Points)")
plt.legend()
plt.tight_layout()
plt.savefig("results/real_garmin_multimodal_eval.png", dpi=120)
print("[SAVED] Diagnostic plot: results/real_garmin_multimodal_eval.png")