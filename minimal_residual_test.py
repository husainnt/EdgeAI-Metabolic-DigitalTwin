import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)
np.random.seed(42)

# ----------------------------------------------------------------------
# 1. Load real patient 1031 CGM data
# ----------------------------------------------------------------------
CSV_PATH = "results/patient_1031_real_cgm.csv"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Could not find {CSV_PATH}. Run this script from your D:\\FYP\\CODE "
        f"directory (or wherever results/patient_1031_real_cgm.csv lives)."
    )

df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"Loaded {len(df)} real CGM readings for patient 1031")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

glucose = df["glucose_mg_dl"].values.astype(np.float32)

# ----------------------------------------------------------------------
# 2. Build a crude "mechanistic baseline" stand-in
# ----------------------------------------------------------------------
WINDOW = 24  # 24 * 5min = 2 hour smoothing window
baseline = pd.Series(glucose).rolling(WINDOW, center=False, min_periods=1).mean().values

residual = glucose - baseline

print(f"\nBaseline (smoothed) glucose stats: mean={baseline.mean():.1f}, std={baseline.std():.1f}")
print(f"Residual (real - baseline) stats: mean={residual.mean():.2f}, std={residual.std():.2f}, "
      f"min={residual.min():.1f}, max={residual.max():.1f}")

# ----------------------------------------------------------------------
# 3. Build sliding-window sequences: past glucose history -> next residual
# ----------------------------------------------------------------------
LOOKBACK = 12   # 12 * 5min = 1 hour of history
HORIZON = 6     # predict 6 steps ahead = 30 minutes ahead

def build_sequences(glucose_arr, baseline_arr, residual_arr, lookback, horizon):
    X, y_residual, y_baseline_target, y_real_target = [], [], [], []
    n = len(glucose_arr)
    for i in range(lookback, n - horizon):
        X.append(glucose_arr[i - lookback:i])
        y_residual.append(residual_arr[i + horizon])
        y_baseline_target.append(baseline_arr[i + horizon])
        y_real_target.append(glucose_arr[i + horizon])
    return (np.array(X), np.array(y_residual),
            np.array(y_baseline_target), np.array(y_real_target))

X, y_res, y_base, y_real = build_sequences(glucose, baseline, residual, LOOKBACK, HORIZON)
print(f"\nBuilt {len(X)} sliding-window samples "
      f"(lookback={LOOKBACK*5}min, horizon={HORIZON*5}min ahead)")

# ----------------------------------------------------------------------
# 4. Train/test split - chronological, not random
# ----------------------------------------------------------------------
split_idx = int(len(X) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_res_train, y_res_test = y_res[:split_idx], y_res[split_idx:]
y_base_train, y_base_test = y_base[:split_idx], y_base[split_idx:]
y_real_train, y_real_test = y_real[:split_idx], y_real[split_idx:]

print(f"Train samples: {len(X_train)} (~{len(X_train)*5/60/24:.1f} days)")
print(f"Test samples:  {len(X_test)} (~{len(X_test)*5/60/24:.1f} days)")

x_mean, x_std = X_train.mean(), X_train.std()
X_train_norm = (X_train - x_mean) / x_std
X_test_norm = (X_test - x_mean) / x_std

# ----------------------------------------------------------------------
# 5. Simulate SPARSE calibration
# ----------------------------------------------------------------------
N_CALIBRATION_POINTS = 80

if len(X_train) > N_CALIBRATION_POINTS:
    calib_idx = np.sort(np.random.choice(len(X_train), N_CALIBRATION_POINTS, replace=False))
else:
    calib_idx = np.arange(len(X_train))

X_calib = X_train_norm[calib_idx]
y_calib = y_res_train[calib_idx]
print(f"\nUsing only {len(X_calib)} sparse calibration points to train the residual model "
      f"(simulating real-world glucometer calibration constraint)")

# ----------------------------------------------------------------------
# 6. Minimal LSTM residual model
# ----------------------------------------------------------------------
class MinimalResidualLSTM(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n.squeeze(0)).squeeze(-1)

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(SeqDataset(X_calib, y_calib), batch_size=8, shuffle=True)

model = MinimalResidualLSTM(hidden_dim=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

EPOCHS = 100
print(f"\nTraining minimal LSTM on {len(X_calib)} sparse calibration points for {EPOCHS} epochs...")
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS} - train MSE loss: {epoch_loss/len(train_loader):.2f}")

# ----------------------------------------------------------------------
# 7. Evaluate on held-out test set
# ----------------------------------------------------------------------
model.eval()
with torch.no_grad():
    X_test_t = torch.tensor(X_test_norm, dtype=torch.float32)
    pred_residual = model(X_test_t).numpy()

pred_final = y_base_test + pred_residual

# ----------------------------------------------------------------------
# 8. Compare baseline-only, LSTM-corrected, and naive persistence
# ----------------------------------------------------------------------
naive_persistence = X_test[:, -1]

rmse_baseline_only = np.sqrt(np.mean((y_real_test - y_base_test) ** 2))
rmse_lstm_corrected = np.sqrt(np.mean((y_real_test - pred_final) ** 2))
rmse_naive_persistence = np.sqrt(np.mean((y_real_test - naive_persistence) ** 2))

mae_baseline_only = np.mean(np.abs(y_real_test - y_base_test))
mae_lstm_corrected = np.mean(np.abs(y_real_test - pred_final))
mae_naive_persistence = np.mean(np.abs(y_real_test - naive_persistence))

print("\n" + "=" * 60)
print("RESULTS - Held-out test set (never seen during training)")
print("=" * 60)
print(f"{'Method':<30} {'RMSE (mg/dL)':<15} {'MAE (mg/dL)':<15}")
print(f"{'-'*60}")
print(f"{'Baseline only (no ML)':<30} {rmse_baseline_only:<15.2f} {mae_baseline_only:<15.2f}")
print(f"{'Naive persistence':<30} {rmse_naive_persistence:<15.2f} {mae_naive_persistence:<15.2f}")
print(f"{'Baseline + LSTM residual':<30} {rmse_lstm_corrected:<15.2f} {mae_lstm_corrected:<15.2f}")
print("=" * 60)

improvement_vs_baseline = ((rmse_baseline_only - rmse_lstm_corrected) / rmse_baseline_only) * 100
improvement_vs_naive = ((rmse_naive_persistence - rmse_lstm_corrected) / rmse_naive_persistence) * 100

print(f"\nLSTM improvement over baseline-only: {improvement_vs_baseline:+.1f}%")
print(f"LSTM improvement over naive persistence: {improvement_vs_naive:+.1f}%")

if rmse_lstm_corrected < rmse_baseline_only and rmse_lstm_corrected < rmse_naive_persistence:
    print("\n[SIGNAL FOUND] The LSTM residual beats BOTH baselines even with only "
          f"{len(X_calib)} sparse calibration points. This is encouraging evidence "
          "that the residual-learning premise is viable - worth proceeding to the "
          "full multimodal architecture.")
elif rmse_lstm_corrected < rmse_baseline_only:
    print("\n[PARTIAL SIGNAL] The LSTM beats the smoothed baseline but not naive "
          "persistence. This is common for short-horizon glucose forecasting since "
          "glucose autocorrelates strongly - worth testing longer horizons or adding "
          "real HR/activity/diet features before concluding either way.")
else:
    print("\n[NO CLEAR SIGNAL YET] The LSTM did not beat either baseline with this "
          "minimal single-signal setup. This does NOT mean the full project premise "
          "fails - it means glucose history alone, with this few calibration points, "
          "isn't enough. Next step: add real HR/activity features (not just glucose "
          "history) before concluding the residual is unlearnable.")

# ----------------------------------------------------------------------
# 9. Save results
# ----------------------------------------------------------------------
os.makedirs("results", exist_ok=True)
summary = {
    "patient_id": 1031,
    "n_calibration_points": int(len(X_calib)),
    "lookback_minutes": LOOKBACK * 5,
    "horizon_minutes": HORIZON * 5,
    "rmse_baseline_only": round(float(rmse_baseline_only), 2),
    "rmse_naive_persistence": round(float(rmse_naive_persistence), 2),
    "rmse_lstm_corrected": round(float(rmse_lstm_corrected), 2),
    "mae_baseline_only": round(float(mae_baseline_only), 2),
    "mae_naive_persistence": round(float(mae_naive_persistence), 2),
    "mae_lstm_corrected": round(float(mae_lstm_corrected), 2),
    "improvement_vs_baseline_pct": round(float(improvement_vs_baseline), 1),
    "improvement_vs_naive_pct": round(float(improvement_vs_naive), 1),
}
import json
with open("results/task9_minimal_residual_test.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n[SAVED] results/task9_minimal_residual_test.json")