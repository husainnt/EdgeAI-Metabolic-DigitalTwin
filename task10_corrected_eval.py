"""
Calibration Density Comparison - Patient 1031
------------------------------------------------
Runs the SAME honest evaluation methodology, but tests FOUR different
calibration densities (2x/day, 3x/day, 4x/day, 6x/day) in one script,
so we can see directly whether calibration frequency is the actual
variable that determines learnability - rather than guessing from one
density at a time.

This directly tests the original project design assumption: "2-3 weeks
of calibration at ~4 readings/day."
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

CSV_PATH = "results/patient_1031_real_cgm.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Could not find {CSV_PATH}. Run from D:\\FYP\\CODE.")

df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
df = df.sort_values("timestamp").reset_index(drop=True)

actual_days = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 86400
print(f"Loaded {len(df)} real CGM readings for patient 1031, span {actual_days:.1f} days\n")

glucose = df["glucose_mg_dl"].values.astype(np.float32)
df["date_str"] = df["timestamp"].dt.strftime("%Y-%m-%d")

WINDOW = 24
baseline = pd.Series(glucose).rolling(WINDOW, center=False, min_periods=1).mean().values

LOOKBACK = 12
HORIZON = 6

def get_calibration_indices(target_hours):
    calibration_indices = set()
    for date_str, group in df.groupby("date_str"):
        for target_hour in target_hours:
            target_time = pd.to_datetime(f"{date_str} {target_hour:02d}:00:00")
            diffs = (group["timestamp"] - target_time).abs()
            closest_idx = diffs.idxmin()
            if diffs.loc[closest_idx] <= pd.Timedelta(minutes=45):
                calibration_indices.add(closest_idx)
    return sorted(calibration_indices)

def build_features(calibration_indices):
    time_since_calib_series = np.zeros(len(glucose))
    calib_val_series = np.zeros(len(glucose))
    last_val = glucose[0]
    tsc = 0.0
    for i in range(len(glucose)):
        if i in calibration_indices:
            last_val = glucose[i]
            tsc = 0.0
        else:
            tsc += 5.0 / 60.0
        time_since_calib_series[i] = tsc
        calib_val_series[i] = last_val

    X, y_real, y_base, tsc_arr, calib_arr = [], [], [], [], []
    for i in range(LOOKBACK, len(glucose) - HORIZON):
        X.append(glucose[i - LOOKBACK:i])
        y_real.append(glucose[i + HORIZON])
        y_base.append(baseline[i + HORIZON])
        tsc_arr.append(time_since_calib_series[i + HORIZON])
        calib_arr.append(calib_val_series[i + HORIZON])
    return (np.array(X), np.array(y_real), np.array(y_base),
            np.array(tsc_arr, dtype=np.float32), np.array(calib_arr, dtype=np.float32))

class CalibAwareResidualLSTM(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim + 2, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x, tsc, calibval):
        x = x.unsqueeze(-1)
        _, (h_n, _) = self.lstm(x)
        h = h_n.squeeze(0)
        combined = torch.cat([h, tsc.unsqueeze(-1), calibval.unsqueeze(-1)], dim=1)
        return self.head(combined).squeeze(-1)

class CalibDataset(Dataset):
    def __init__(self, X, tsc, calibval, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.tsc = torch.tensor(tsc, dtype=torch.float32)
        self.calibval = torch.tensor(calibval, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.tsc[idx], self.calibval[idx], self.y[idx]

def run_density_experiment(name, target_hours, epochs=100, lr=0.005, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    calib_indices = get_calibration_indices(target_hours)
    X, y_real, y_base, tsc_arr, calib_arr = build_features(calib_indices)

    split_idx = int(len(X) * 0.7)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_real_train, y_real_test = y_real[:split_idx], y_real[split_idx:]
    y_base_train, y_base_test = y_base[:split_idx], y_base[split_idx:]
    tsc_train, tsc_test = tsc_arr[:split_idx], tsc_arr[split_idx:]
    calibval_train, calibval_test = calib_arr[:split_idx], calib_arr[split_idx:]

    x_mean, x_std = X_train.mean(), X_train.std()
    X_train_n, X_test_n = (X_train - x_mean) / x_std, (X_test - x_mean) / x_std

    tsc_mean, tsc_std = tsc_train.mean(), tsc_train.std() + 1e-6
    tsc_train_n, tsc_test_n = (tsc_train - tsc_mean) / tsc_std, (tsc_test - tsc_mean) / tsc_std

    cv_mean, cv_std = calibval_train.mean(), calibval_train.std() + 1e-6
    cv_train_n, cv_test_n = (calibval_train - cv_mean) / cv_std, (calibval_test - cv_mean) / cv_std

    n_calib_points = len(calib_indices)
    n_train_calib = min(len(X_train), n_calib_points)
    calib_idx = np.sort(np.random.choice(len(X_train), n_train_calib, replace=False))

    X_c = X_train_n[calib_idx]
    y_c = y_real_train[calib_idx] - y_base_train[calib_idx]
    tsc_c = tsc_train_n[calib_idx]
    cv_c = cv_train_n[calib_idx]

    loader = DataLoader(CalibDataset(X_c, tsc_c, cv_c, y_c), batch_size=8, shuffle=True)
    model = CalibAwareResidualLSTM(hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    final_loss = None
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, tscb, cvb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb, tscb, cvb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        final_loss = epoch_loss / len(loader)

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test_n, dtype=torch.float32)
        tsc_test_t = torch.tensor(tsc_test_n, dtype=torch.float32)
        cv_test_t = torch.tensor(cv_test_n, dtype=torch.float32)
        pred_res = model(X_test_t, tsc_test_t, cv_test_t).numpy()

    pred_final = y_base_test + pred_res
    rmse_base = np.sqrt(np.mean((y_real_test - y_base_test) ** 2))
    rmse_hyb = np.sqrt(np.mean((y_real_test - pred_final) ** 2))
    mae_base = np.mean(np.abs(y_real_test - y_base_test))
    mae_hyb = np.mean(np.abs(y_real_test - pred_final))
    improvement_pct = ((rmse_base - rmse_hyb) / rmse_base) * 100

    print(f"[{name}] n_calib={n_calib_points:3d} | Baseline RMSE={rmse_base:6.2f} | "
          f"Hybrid RMSE={rmse_hyb:6.2f} | Improvement={improvement_pct:+6.1f}% | final_train_loss={final_loss:.1f}")

    return {
        "name": name,
        "n_calibration_points": n_calib_points,
        "rmse_baseline": round(float(rmse_base), 2),
        "rmse_hybrid": round(float(rmse_hyb), 2),
        "mae_baseline": round(float(mae_base), 2),
        "mae_hybrid": round(float(mae_hyb), 2),
        "improvement_pct": round(float(improvement_pct), 1),
    }

print("Running calibration-density comparison (tests our original '4x/day' design assumption)...")
print("=" * 100)

experiments = [
    ("2x_per_day", [8, 20]),
    ("3x_per_day", [8, 14, 20]),
    ("4x_per_day", [7, 12, 17, 22]),
    ("6x_per_day", [6, 10, 14, 18, 21, 23]),
]

results = []
for name, hours in experiments:
    results.append(run_density_experiment(name, hours))

print("=" * 100)
print("\nSUMMARY TABLE")
print(f"{'Density':<15}{'#Calib Pts':<12}{'Baseline RMSE':<16}{'Hybrid RMSE':<14}{'Improvement':<12}")
for r in results:
    print(f"{r['name']:<15}{r['n_calibration_points']:<12}{r['rmse_baseline']:<16}{r['rmse_hybrid']:<14}{r['improvement_pct']:+.1f}%")

os.makedirs("results", exist_ok=True)
with open("results/task10_density_comparison.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n[SAVED] results/task10_density_comparison.json")

plt.figure(figsize=(8, 5))
names = [r["name"] for r in results]
improvements = [r["improvement_pct"] for r in results]
n_points = [r["n_calibration_points"] for r in results]
plt.bar(names, improvements, color=["red" if i < 0 else "green" for i in improvements])
plt.axhline(0, color="black", linewidth=0.8)
plt.ylabel("RMSE Improvement over Baseline (%)")
plt.title("Does Calibration Frequency Determine Learnability? (Patient 1031)")
for i, (n, imp) in enumerate(zip(n_points, improvements)):
    plt.text(i, imp, f"n={n}", ha="center", va="bottom" if imp >= 0 else "top")
plt.tight_layout()
plt.savefig("results/task10_density_comparison_plot.png", dpi=120)
print("[SAVED] results/task10_density_comparison_plot.png")