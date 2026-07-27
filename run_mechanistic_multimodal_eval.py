"""
Option B: Mechanistic ODE Baseline + Garmin HR Multimodal Hybrid Twin
-----------------------------------------------------------------------
1. Replaces flat forward-fill with an Open-Loop T2D ODE Minimal Model (y_ODE).
2. Uses real Garmin HR telemetry + sparse calibration state to predict y_real - y_ODE.
3. Evaluates 100-epoch convergence on Patient 1031.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Set deterministic random seeds
torch.manual_seed(42)
np.random.seed(42)

CSV_PATH = "results/patient_1031_real_cgm_with_hr.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Could not find {CSV_PATH}. Run merge script first.")

# Step 1: Load aligned patient dataset
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

# Step 2: Extract 2x/day sparse calibration points (~8:00 AM & 8:00 PM)
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

# Step 3: Compute Open-Loop T2D ODE Simulation (Mechanistic Baseline)
# Bergman Minimal Model Equations:
# dG/dt = -p1*(G - Gb) - X*G
# dX/dt = -p2*X + p3*(I - Ib)
p1, p2, Gb = 0.008, 0.02, 110.0  # Calibrated T2D metabolic rate constants

def t2d_minimal_ode(t, y):
    G, X = y[0], y[1]
    dGdt = -p1 * (G - Gb) - X * G
    dXdt = -p2 * X
    return [dGdt, dXdt]

ode_baseline = np.zeros(len(glucose), dtype=np.float32)
time_since_calib_series = np.zeros(len(glucose), dtype=np.float32)
calib_val_series = np.zeros(len(glucose), dtype=np.float32)

current_g0 = glucose[0]
current_t0 = 0.0

for i in range(len(glucose)):
    if i in calibration_indices:
        current_g0 = glucose[i]
        current_t0 = 0.0

    # Integrate ODE forward from last calibration point
    t_span = (0.0, current_t0 if current_t0 > 0 else 0.001)
    sol = solve_ivp(t2d_minimal_ode, t_span, [current_g0, 0.0], method='RK45')
    
    ode_baseline[i] = sol.y[0][-1]
    time_since_calib_series[i] = current_t0
    calib_val_series[i] = current_g0
    current_t0 += 5.0 / 60.0  # 5-minute step in hours

# Step 4: Construct Dataset Windows [HR Sequence + Sparse ODE Context]
LOOKBACK = 12  # 60 mins of Garmin HR history
HORIZON = 6    # 30 mins ahead forecast

X_hr_seq, y_real, y_ode, tsc_arr, calib_arr = [], [], [], [], []

for i in range(LOOKBACK, len(glucose) - HORIZON):
    X_hr_seq.append(hr_data[i - LOOKBACK:i])
    y_real.append(glucose[i + HORIZON])
    y_ode.append(ode_baseline[i + HORIZON])
    tsc_arr.append(time_since_calib_series[i + HORIZON])
    calib_arr.append(calib_val_series[i + HORIZON])

X_hr_seq = np.array(X_hr_seq, dtype=np.float32)
y_real = np.array(y_real, dtype=np.float32)
y_ode = np.array(y_ode, dtype=np.float32)
tsc_arr = np.array(tsc_arr, dtype=np.float32)
calib_arr = np.array(calib_arr, dtype=np.float32)

# Chronological Split (70% Train / 30% Test)
split_idx = int(len(X_hr_seq) * 0.7)

X_hr_train, X_hr_test = X_hr_seq[:split_idx], X_hr_seq[split_idx:]
y_real_train, y_real_test = y_real[:split_idx], y_real[split_idx:]
y_ode_train, y_ode_test = y_ode[:split_idx], y_ode[split_idx:]
tsc_train, tsc_test = tsc_arr[:split_idx], tsc_arr[split_idx:]
calibval_train, calibval_test = calib_arr[:split_idx], calib_arr[split_idx:]

# Feature Normalization
hr_mean, hr_std = X_hr_train.mean(), X_hr_train.std() + 1e-6
X_hr_tr_n = (X_hr_train - hr_mean) / hr_std
X_hr_te_n = (X_hr_test - hr_mean) / hr_std

tsc_mean, tsc_std = tsc_train.mean(), tsc_train.std() + 1e-6
tsc_tr_n = (tsc_train - tsc_mean) / tsc_std
tsc_te_n = (tsc_test - tsc_mean) / tsc_std

cv_mean, cv_std = calibval_train.mean(), calibval_train.std() + 1e-6
cv_tr_n = (calibval_train - cv_mean) / cv_std
cv_te_n = (calibval_test - cv_mean) / cv_std

# ML Target: Residual between Ground Truth and Mechanistic ODE prediction
y_train_residual = y_real_train - y_ode_train

# Step 5: PyTorch Dataset & Residual Model
class MechanisticMultimodalDataset(Dataset):
    def __init__(self, X_hr, tsc, calibval, y):
        self.X_hr = torch.tensor(X_hr, dtype=torch.float32)
        self.tsc = torch.tensor(tsc, dtype=torch.float32)
        self.calibval = torch.tensor(calibval, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X_hr)
    def __getitem__(self, idx): return self.X_hr[idx], self.tsc[idx], self.calibval[idx], self.y[idx]

class MechanisticResidualLSTM(nn.Module):
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

loader = DataLoader(
    MechanisticMultimodalDataset(X_hr_tr_n, tsc_tr_n, cv_tr_n, y_train_residual), 
    batch_size=32, shuffle=True
)

EPOCHS = 100
model = MechanisticResidualLSTM(hidden_dim=16)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
loss_fn = nn.MSELoss()

print("\nTraining Mechanistic Multimodal Hybrid Model (100 Epochs)...")
print("=" * 85)

model.train()
for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    for xb_hr, tscb, cvb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb_hr, tscb, cvb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    scheduler.step()
    if epoch % 10 == 0:
        avg_loss = epoch_loss / len(loader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:03d}/{EPOCHS} | Train MSE Loss: {avg_loss:.2f} | LR: {current_lr:.6f}")

# Step 6: Evaluation
model.eval()
with torch.no_grad():
    X_hr_test_t = torch.tensor(X_hr_te_n, dtype=torch.float32)
    tsc_test_t = torch.tensor(tsc_te_n, dtype=torch.float32)
    cv_test_t = torch.tensor(cv_te_n, dtype=torch.float32)
    pred_res = model(X_hr_test_t, tsc_test_t, cv_test_t).numpy()

pred_final = y_ode_test + pred_res

rmse_ode = np.sqrt(np.mean((y_real_test - y_ode_test) ** 2))
rmse_hybrid = np.sqrt(np.mean((y_real_test - pred_final) ** 2))
mae_ode = np.mean(np.abs(y_real_test - y_ode_test))
mae_hybrid = np.mean(np.abs(y_real_test - pred_final))
improvement_pct = ((rmse_ode - rmse_hybrid) / rmse_ode) * 100

print("=" * 85)
print(f"MECHANISTIC HYBRID RESULTS (Option B): Patient 1031 ({actual_days:.1f} Days)")
print("=" * 85)
print(f"  • Mechanistic ODE Baseline RMSE:   {rmse_ode:.2f} mg/dL | MAE: {mae_ode:.2f} mg/dL")
print(f"  • Mechanistic Hybrid Twin RMSE:    {rmse_hybrid:.2f} mg/dL | MAE: {mae_hybrid:.2f} mg/dL")
print(f"  • RMSE Improvement over Physics:   {improvement_pct:+.1f}%")
print("=" * 85)

# Step 7: Plotting
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(14, 6))
plot_n = min(300, len(y_real_test))
plt.plot(y_real_test[:plot_n], label="Real Glucose (Ground Truth)", color="black", linewidth=1.5)
plt.plot(y_ode_test[:plot_n], label="Mechanistic ODE Baseline (Physiological Drift)", color="blue", linestyle="--", alpha=0.7)
plt.plot(pred_final[:plot_n], label="Full Hybrid Twin (ODE + Garmin HR ML)", color="green", alpha=0.85)
plt.xlabel("Test Sample Index (Chronological)")
plt.ylabel("Glucose (mg/dL)")
plt.title(f"Option B: Mechanistic ODE vs. Full Hybrid Twin — Patient 1031")
plt.legend()
plt.tight_layout()
plt.savefig("results/mechanistic_hybrid_eval.png", dpi=120)
print("[SAVED] Diagnostic plot: results/mechanistic_hybrid_eval.png")