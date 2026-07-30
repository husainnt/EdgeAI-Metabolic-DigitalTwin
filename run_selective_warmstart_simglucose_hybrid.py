"""
Selective State Warm-Started simglucose (UVa/Padova T2D) + Garmin HR Hybrid
---------------------------------------------------------------------------
1. Instantiates simglucose T1DPatient ("adult#001") with calibrated T2D parameters:
   - Vmx * 0.75, kp3 * 0.70.
2. For each sparse calibration point (~8am/8pm):
   - Computes Vg = patient.state[3] / Gb.
   - Warm-starts active glucose & insulin compartments [3, 4, 5, 7, 8, 9, 10, 11, 12] IN-PLACE.
   - Steps patient.step() open-loop using T2DPancreaticController basal policy.
3. Includes an explicit Parameter Reset Guard to verify/enforce T2D parameter survival across resets.
4. Trains Multimodal Residual LSTM (Garmin HR sequence + calibration context) for 100 epochs.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Ensure simglucose modules are discoverable
sys.path.append(os.path.abspath("SIM-GLUCOSE"))

from simglucose.patient.t1dpatient import T1DPatient, Action as PatientAction

# Set deterministic random seeds
torch.manual_seed(42)
np.random.seed(42)


# --- 1. Calibrated T2D Pancreatic Controller ---
class T2DPancreaticController:
    def __init__(self, gb, basal_rate, kp=0.0005, max_secretion=0.05):
        self.gb = gb
        self.basal_rate = basal_rate
        self.kp = kp
        self.max_secretion = max_secretion

    def basal_for(self, bg):
        if bg <= self.gb:
            secretion = self.basal_rate
        else:
            secretion = self.basal_rate + self.kp * (bg - self.gb)
        return min(secretion, self.max_secretion)


# --- 2. Patient Builder & Warm-Start Utilities ---
def build_calibrated_t2d_patient():
    patient = T1DPatient.withName("adult#001")
    params = patient._params
    params['Vmx'] *= 0.75
    params['kp3'] *= 0.70
    patient._params = params
    patient.reset()
    return patient

def get_Vg(patient):
    # Plasma distribution volume Vg = Gp_rest / Gb
    return patient.state[3] / patient._params['Gb']

def warm_start(patient, target_bg):
    """
    Proportionally rescales active glucose and insulin compartments 
    [3, 4, 5, 7, 8, 9, 10, 11, 12] in-place to match target_bg, 
    keeping meal compartments [0, 1, 2] and insulin deviation [6] at 0.
    """
    Vg = get_Vg(patient)
    current_bg = patient.state[3] / Vg
    scale = target_bg / current_bg
    
    # Mutate numpy array elements in-place (bypasses read-only property getter)
    for i in [3, 4, 5, 7, 8, 9, 10, 11, 12]:
        patient.state[i] *= scale

def simulate_open_loop(patient, controller, n_steps_5min):
    """
    Steps patient directly open-loop for n_steps_5min intervals using PatientAction.
    """
    Vg = get_Vg(patient)
    traj = []
    sub_steps_per_5min = max(1, int(round(5.0 / patient.sample_time)))
    for _ in range(n_steps_5min):
        for _ in range(sub_steps_per_5min):
            bg_now = patient.state[3] / Vg
            basal = controller.basal_for(bg_now)
            
            # Direct patient step using patient-level action signature (CHO, insulin)
            patient.step(PatientAction(CHO=0, insulin=basal))
                
        traj.append(patient.state[3] / Vg)
    return traj


# --- 3. Load Patient 1031 Dataset ---
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


# --- 4. Extract 2x/day Calibration Indices (~8am/8pm) ---
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


# --- 5. Generate Selective Warm-Started simglucose Baseline + PARAMETER RESET GUARD ---
patient = build_calibrated_t2d_patient()
params = patient._params
basal_rate_correct = params['u2ss'] * params['BW'] / 6000
controller = T2DPancreaticController(gb=params['Gb'], basal_rate=basal_rate_correct)

sim_baseline = np.zeros(len(glucose), dtype=np.float32)
time_since_calib_series = np.zeros(len(glucose), dtype=np.float32)
calib_val_series = np.zeros(len(glucose), dtype=np.float32)

print("Simulating open-loop simglucose ODE trajectories with selective 13-compartment in-place warm-starting...")

for k in range(len(calibration_indices)):
    idx_start = calibration_indices[k]
    idx_end = calibration_indices[k+1] if k + 1 < len(calibration_indices) else len(glucose)
    n_steps = idx_end - idx_start
    
    g_calib = glucose[idx_start]
    
    # Store parameters before reset to verify persistence
    vmx_before = patient._params['Vmx']
    kp3_before = patient._params['kp3']
    
    # Execute patient reset
    patient.reset()
    
    vmx_after = patient._params['Vmx']
    kp3_after = patient._params['kp3']
    
    # Debug Check & Parameter Guard on Segment 0
    if k == 0:
        print("\n" + "=" * 65)
        print(" [DEBUG CHECK] SIMGLUCOSE PATIENT RESET PARAMETER SURVIVAL TEST")
        print("=" * 65)
        print(f"  • Vmx BEFORE reset: {vmx_before:.8f}")
        print(f"  • Vmx AFTER reset:  {vmx_after:.8f}  (Expected: ~0.02348925)")
        print(f"  • kp3 AFTER reset:  {kp3_after:.8f}  (Expected: ~0.00763000)")
        if vmx_before == vmx_after and kp3_before == kp3_after:
            print("  ✓ SUCCESS: Custom T2D parameters survived patient.reset() intact!")
        else:
            print("  ❌ CRITICAL WARNING: patient.reset() wiped custom parameters!")
            print("     Re-applying T2D modifications manually after reset...")
            patient._params['Vmx'] = vmx_before
            patient._params['kp3'] = kp3_before
        print("=" * 65 + "\n")
    elif vmx_before != vmx_after or kp3_before != kp3_after:
        # Enforce persistence on subsequent segments if reset wipes them
        patient._params['Vmx'] = vmx_before
        patient._params['kp3'] = kp3_before

    # Warm-start patient state to fingerstick value
    warm_start(patient, g_calib)
    
    # Step ODE model forward
    traj = simulate_open_loop(patient, controller, n_steps)
    
    for s_i, step_idx in enumerate(range(idx_start, idx_end)):
        sim_baseline[step_idx] = traj[s_i]
        time_since_calib_series[step_idx] = (s_i * 5.0) / 60.0
        calib_val_series[step_idx] = g_calib

# Handle initial prefix before first calibration point
for i in range(calibration_indices[0]):
    sim_baseline[i] = glucose[0]
    calib_val_series[i] = glucose[0]
    time_since_calib_series[i] = (i * 5.0) / 60.0

print(f"✓ Warm-started simglucose simulation complete! Baseline Mean: {sim_baseline.mean():.1f} mg/dL")


# --- 6. Dataset Windowing & Chronological Train/Test Split ---
LOOKBACK = 12  # 60 mins of Garmin HR history
HORIZON = 6    # 30 mins ahead forecast

X_hr_seq, y_real, y_sim, tsc_arr, calib_arr = [], [], [], [], []

for i in range(LOOKBACK, len(glucose) - HORIZON):
    X_hr_seq.append(hr_data[i - LOOKBACK:i])
    y_real.append(glucose[i + HORIZON])
    y_sim.append(sim_baseline[i + HORIZON])
    tsc_arr.append(time_since_calib_series[i + HORIZON])
    calib_arr.append(calib_val_series[i + HORIZON])

X_hr_seq = np.array(X_hr_seq, dtype=np.float32)
y_real = np.array(y_real, dtype=np.float32)
y_sim = np.array(y_sim, dtype=np.float32)
tsc_arr = np.array(tsc_arr, dtype=np.float32)
calib_arr = np.array(calib_arr, dtype=np.float32)

split_idx = int(len(X_hr_seq) * 0.7)

X_hr_train, X_hr_test = X_hr_seq[:split_idx], X_hr_seq[split_idx:]
y_real_train, y_real_test = y_real[:split_idx], y_real[split_idx:]
y_sim_train, y_sim_test = y_sim[:split_idx], y_sim[split_idx:]
tsc_train, tsc_test = tsc_arr[:split_idx], tsc_arr[split_idx:]
calibval_train, calibval_test = calib_arr[:split_idx], calib_arr[split_idx:]

# Normalization
hr_mean, hr_std = X_hr_train.mean(), X_hr_train.std() + 1e-6
X_hr_tr_n = (X_hr_train - hr_mean) / hr_std
X_hr_te_n = (X_hr_test - hr_mean) / hr_std

tsc_mean, tsc_std = tsc_train.mean(), tsc_train.std() + 1e-6
tsc_tr_n = (tsc_train - tsc_mean) / tsc_std
tsc_te_n = (tsc_test - tsc_mean) / tsc_std

cv_mean, cv_std = calibval_train.mean(), calibval_train.std() + 1e-6
cv_tr_n = (calibval_train - cv_mean) / cv_std
cv_te_n = (calibval_test - cv_mean) / cv_std

# Target for ML Head: Residual between Ground Truth and simglucose ODE
y_train_residual = y_real_train - y_sim_train


# --- 7. PyTorch Multimodal Residual Model ---
class SelectiveWarmStartDataset(Dataset):
    def __init__(self, X_hr, tsc, calibval, y):
        self.X_hr = torch.tensor(X_hr, dtype=torch.float32)
        self.tsc = torch.tensor(tsc, dtype=torch.float32)
        self.calibval = torch.tensor(calibval, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X_hr)
    def __getitem__(self, idx): return self.X_hr[idx], self.tsc[idx], self.calibval[idx], self.y[idx]

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

loader = DataLoader(
    SelectiveWarmStartDataset(X_hr_tr_n, tsc_tr_n, cv_tr_n, y_train_residual), 
    batch_size=32, shuffle=True
)


# --- 8. Model Training (100 Epochs + Cosine Scheduler) ---
EPOCHS = 100
model = SelectiveWarmStartResidualLSTM(hidden_dim=16)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
loss_fn = nn.MSELoss()

print("\nTraining Residual LSTM on Selective Warm-Start simglucose Residuals (100 Epochs)...")
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


# --- 9. Held-Out Test Evaluation ---
model.eval()
with torch.no_grad():
    X_hr_test_t = torch.tensor(X_hr_te_n, dtype=torch.float32)
    tsc_test_t = torch.tensor(tsc_te_n, dtype=torch.float32)
    cv_test_t = torch.tensor(cv_te_n, dtype=torch.float32)
    pred_res = model(X_hr_test_t, tsc_test_t, cv_test_t).numpy()

pred_final = y_sim_test + pred_res

rmse_sim = np.sqrt(np.mean((y_real_test - y_sim_test) ** 2))
rmse_hybrid = np.sqrt(np.mean((y_real_test - pred_final) ** 2))
mae_sim = np.mean(np.abs(y_real_test - y_sim_test))
mae_hybrid = np.mean(np.abs(y_real_test - pred_final))
improvement_pct = ((rmse_sim - rmse_hybrid) / rmse_sim) * 100

print("=" * 85)
print(f"SELECTIVE WARM-START SIMGLUCOSE HYBRID RESULTS: Patient 1031 ({actual_days:.1f} Days)")
print("=" * 85)
print(f"  • simglucose T2D Baseline RMSE: {rmse_sim:.2f} mg/dL | MAE: {mae_sim:.2f} mg/dL")
print(f"  • True Hybrid Twin RMSE:        {rmse_hybrid:.2f} mg/dL | MAE: {mae_hybrid:.2f} mg/dL")
print(f"  • Improvement over Physics:    {improvement_pct:+.1f}%")
print("=" * 85)


# --- 10. Diagnostic Plot ---
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(14, 6))
plot_n = min(300, len(y_real_test))
plt.plot(y_real_test[:plot_n], label="Real Glucose (Ground Truth)", color="black", linewidth=1.5)
plt.plot(y_sim_test[:plot_n], label="simglucose T2D Baseline (Selective Warm-Start adult#001)", color="blue", linestyle="--", alpha=0.7)
plt.plot(pred_final[:plot_n], label="True Hybrid Twin (simglucose + Garmin HR ML)", color="green", alpha=0.85)
plt.xlabel("Test Sample Index (Chronological)")
plt.ylabel("Glucose (mg/dL)")
plt.title("Option B Final: Selective Warm-Start simglucose T2D Engine vs. Full Hybrid Twin")
plt.legend()
plt.tight_layout()
plt.savefig("results/selective_warmstart_simglucose_hybrid_eval.png", dpi=120)
print("[SAVED] Diagnostic plot: results/selective_warmstart_simglucose_hybrid_eval.png")