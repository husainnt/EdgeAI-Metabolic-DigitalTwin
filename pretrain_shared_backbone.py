"""
Shared Backbone Pretraining
============================
Pools calibration-window residual samples across the full extracted cohort
(results/cohort/patient_<pid>_merged.csv) and trains ONE shared HR+Sleep
residual model -- this is the real answer to the panel's personalization
objection: population pretraining first, then a small per-patient residual
head fine-tune (see finetune_patient.py).

Methodology matches the validated headline pipeline exactly:
    - Same T2DPancreaticController / build_calibrated_t2d_patient / warm_start
      / simulate_open_loop from run_selective_warmstart_simglucose_hybrid.py
    - Same LOOKBACK=12 (60 min), HORIZON=6 (30 min) windowing
    - Single-scalar +30-min-exact target (Methodology Correction #1 --
      NOT the flawed 6-step vector-target approach)
    - Same 2x/day (~8am/8pm, 45-min tolerance) calibration extraction

Split strategy: each patient's own data is split 70/30 chronologically
(train/test) exactly as before. All patients' TRAIN portions are pooled
to fit the shared backbone. Each patient's TEST portion is held out
entirely for the fine-tuning stage's evaluation -- no leakage.

NOTE ON RUNTIME: the simglucose warm-start ODE simulation is the
expensive part (stepped per 5-min interval per patient). Test on a small
subset first (see LIMIT_PATIENTS below) before committing to a full
577-patient run, which may take a long time depending on your machine.
"""

import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd

from run_selective_warmstart_simglucose_hybrid import (
    build_calibrated_t2d_patient,
    warm_start,
    simulate_open_loop,
    T2DPancreaticController
)

COHORT_DIR = "results/cohort"
LOOKBACK = 12
HORIZON = 6
OUTPUT_MODEL_PATH = "shared_pretrained_backbone.pt"
OUTPUT_NORM_STATS_PATH = "shared_pretrain_norm_stats.npz"

# Set to a small number (e.g. 20) for a smoke test before running the full cohort.
# Set to None to run on the entire extracted cohort.
LIMIT_PATIENTS = 20

torch.manual_seed(42)
np.random.seed(42)


class SharedHRSleepResidualLSTM(nn.Module):
    """
    Shared backbone: independent HR and Sleep LSTM encoders + context MLP,
    fused into a residual head. The encoders + context arm are the
    population-pretrained "backbone"; the fusion_head is the small piece
    that gets fine-tuned per patient in finetune_patient.py.
    """
    def __init__(self, hidden_dim=32, context_dim=2):
        super().__init__()
        self.hr_encoder = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.sleep_encoder = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.fc_context = nn.Linear(context_dim, 16)
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.relu = nn.ReLU()

    def forward(self, hr_seq, sleep_seq, ctx):
        hr_seq = hr_seq.unsqueeze(-1)
        sleep_seq = sleep_seq.unsqueeze(-1)
        _, (h_hr, _) = self.hr_encoder(hr_seq)
        _, (h_sl, _) = self.sleep_encoder(sleep_seq)
        ctx_feat = self.relu(self.fc_context(ctx))
        combined = torch.cat([h_hr[-1], h_sl[-1], ctx_feat], dim=1)
        return self.fusion_head(combined).squeeze(-1)


class PooledDataset(Dataset):
    def __init__(self, X_hr, X_sleep, X_ctx, y):
        self.X_hr = torch.tensor(X_hr, dtype=torch.float32)
        self.X_sleep = torch.tensor(X_sleep, dtype=torch.float32)
        self.X_ctx = torch.tensor(X_ctx, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_hr[idx], self.X_sleep[idx], self.X_ctx[idx], self.y[idx]


def extract_2x_daily_calibrations(df, window_tolerance_min=45):
    timestamps_naive = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    df["timestamp_naive"] = timestamps_naive
    dates = timestamps_naive.dt.date.unique()
    calib_indices = []
    for d in dates:
        day_df = df[df["timestamp_naive"].dt.date == d]
        for target_hour in [8, 20]:
            t = pd.to_datetime(f"{d} {target_hour:02d}:00:00")
            diffs = (day_df["timestamp_naive"] - t).abs()
            if not diffs.empty and diffs.min() <= pd.Timedelta(minutes=window_tolerance_min):
                calib_indices.append(diffs.idxmin())
    return sorted(list(set(calib_indices)))


def build_patient_windows(df, calib_indices, patient):
    """
    Builds single-scalar +30-min windows for one patient: HR sequence,
    Sleep sequence, context (tsc, calib_val), true glucose, sim baseline.
    Matches the corrected (Methodology Correction #1) single-scalar pipeline.
    """
    glucose = df["glucose_mg_dl"].values.astype(np.float32)
    hr = df["heart_rate"].values.astype(np.float32)
    sleep = df["sleep_status"].values.astype(np.float32)
    num_samples = len(df)

    if len(calib_indices) < 2:
        return None

    params = patient._params
    basal_rate_correct = params["u2ss"] * params["BW"] / 6000
    controller = T2DPancreaticController(gb=params["Gb"], basal_rate=basal_rate_correct)

    sim_baseline = np.zeros(num_samples, dtype=np.float32)
    tsc_series = np.zeros(num_samples, dtype=np.float32)
    calib_series = np.zeros(num_samples, dtype=np.float32)

    for k in range(len(calib_indices)):
        idx_start = calib_indices[k]
        idx_end = calib_indices[k + 1] if k + 1 < len(calib_indices) else num_samples
        n_steps = idx_end - idx_start
        g_calib = glucose[idx_start]

        patient.reset()
        warm_start(patient, g_calib)
        traj = simulate_open_loop(patient, controller, n_steps)

        for s_i, step_idx in enumerate(range(idx_start, idx_end)):
            sim_baseline[step_idx] = traj[s_i]
            tsc_series[step_idx] = (s_i * 5.0) / 60.0
            calib_series[step_idx] = g_calib

    for i in range(calib_indices[0]):
        sim_baseline[i] = glucose[0]
        calib_series[i] = glucose[0]
        tsc_series[i] = (i * 5.0) / 60.0

    X_hr, X_sleep, y_real, y_sim, tsc_arr, calib_arr = [], [], [], [], [], []
    for i in range(LOOKBACK, num_samples - HORIZON):
        if sim_baseline[i + HORIZON] == 0:
            continue
        X_hr.append(hr[i - LOOKBACK:i])
        X_sleep.append(sleep[i - LOOKBACK:i])
        y_real.append(glucose[i + HORIZON])
        y_sim.append(sim_baseline[i + HORIZON])
        tsc_arr.append(tsc_series[i + HORIZON])
        calib_arr.append(calib_series[i + HORIZON])

    if len(y_real) < 20:
        return None

    return {
        "X_hr": np.array(X_hr, dtype=np.float32),
        "X_sleep": np.array(X_sleep, dtype=np.float32),
        "tsc": np.array(tsc_arr, dtype=np.float32),
        "calib_val": np.array(calib_arr, dtype=np.float32),
        "y_real": np.array(y_real, dtype=np.float32),
        "y_sim": np.array(y_sim, dtype=np.float32),
    }


def run_pretraining():
    csv_files = sorted(glob.glob(os.path.join(COHORT_DIR, "patient_*_merged.csv")))
    if LIMIT_PATIENTS is not None:
        csv_files = csv_files[:LIMIT_PATIENTS]

    print(f"[+] Building pooled dataset from {len(csv_files)} patients...")

    pooled_train_hr, pooled_train_sleep, pooled_train_tsc, pooled_train_calib, pooled_train_y_res = [], [], [], [], []
    held_out_test = {}  # patient_id -> dict of test-split arrays, for later fine-tune eval

    skipped = 0
    for i, csv_path in enumerate(csv_files):
        pid = os.path.basename(csv_path).replace("patient_", "").replace("_merged.csv", "")
        df = pd.read_csv(csv_path)

        try:
            calib_indices = extract_2x_daily_calibrations(df)
            patient = build_calibrated_t2d_patient()
            windows = build_patient_windows(df, calib_indices, patient)
        except Exception as e:
            skipped += 1
            continue

        if windows is None:
            skipped += 1
            continue

        n = len(windows["y_real"])
        split = int(n * 0.7)

        y_train_res = windows["y_real"][:split] - windows["y_sim"][:split]

        pooled_train_hr.append(windows["X_hr"][:split])
        pooled_train_sleep.append(windows["X_sleep"][:split])
        pooled_train_tsc.append(windows["tsc"][:split])
        pooled_train_calib.append(windows["calib_val"][:split])
        pooled_train_y_res.append(y_train_res)

        held_out_test[pid] = {
            "X_hr": windows["X_hr"][split:],
            "X_sleep": windows["X_sleep"][split:],
            "tsc": windows["tsc"][split:],
            "calib_val": windows["calib_val"][split:],
            "y_real": windows["y_real"][split:],
            "y_sim": windows["y_sim"][split:],
            # keep train split too, for fine-tuning stage
            "X_hr_train": windows["X_hr"][:split],
            "X_sleep_train": windows["X_sleep"][:split],
            "tsc_train": windows["tsc"][:split],
            "calib_val_train": windows["calib_val"][:split],
            "y_real_train": windows["y_real"][:split],
            "y_sim_train": windows["y_sim"][:split],
        }

        if (i + 1) % 10 == 0:
            print(f"    ...processed {i + 1}/{len(csv_files)} patients (skipped so far: {skipped})")

    print(f"[+] Pooled {len(pooled_train_y_res)} patients into shared training set (skipped {skipped}).")

    X_hr_all = np.concatenate(pooled_train_hr, axis=0)
    X_sleep_all = np.concatenate(pooled_train_sleep, axis=0)
    tsc_all = np.concatenate(pooled_train_tsc, axis=0)
    calib_all = np.concatenate(pooled_train_calib, axis=0)
    y_res_all = np.concatenate(pooled_train_y_res, axis=0)

    print(f"[+] Total pooled training windows: {len(y_res_all)}")

    # Global normalization stats (fit on pooled TRAIN data only)
    hr_mean, hr_std = X_hr_all.mean(), X_hr_all.std() + 1e-6
    sleep_mean, sleep_std = X_sleep_all.mean(), X_sleep_all.std() + 1e-6
    tsc_mean, tsc_std = tsc_all.mean(), tsc_all.std() + 1e-6
    calib_mean, calib_std = calib_all.mean(), calib_all.std() + 1e-6

    np.savez(
        OUTPUT_NORM_STATS_PATH,
        hr_mean=hr_mean, hr_std=hr_std,
        sleep_mean=sleep_mean, sleep_std=sleep_std,
        tsc_mean=tsc_mean, tsc_std=tsc_std,
        calib_mean=calib_mean, calib_std=calib_std,
    )
    print(f"[+] Saved normalization stats to {OUTPUT_NORM_STATS_PATH}")

    X_hr_n = (X_hr_all - hr_mean) / hr_std
    X_sleep_n = (X_sleep_all - sleep_mean) / sleep_std
    tsc_n = (tsc_all - tsc_mean) / tsc_std
    calib_n = (calib_all - calib_mean) / calib_std
    X_ctx_n = np.stack([tsc_n, calib_n], axis=1)

    loader = DataLoader(
        PooledDataset(X_hr_n, X_sleep_n, X_ctx_n, y_res_all),
        batch_size=64, shuffle=True
    )

    model = SharedHRSleepResidualLSTM(hidden_dim=32, context_dim=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    epochs = 50  # pooled dataset is much larger than single-patient, fewer epochs needed
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.MSELoss()

    print(f"\n[+] Training shared backbone on {len(y_res_all)} pooled windows ({epochs} epochs)...")
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for b_hr, b_sleep, b_ctx, b_y in loader:
            optimizer.zero_grad()
            pred = model(b_hr, b_sleep, b_ctx)
            loss = loss_fn(pred, b_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        if epoch % 5 == 0:
            print(f"    Epoch {epoch:03d}/{epochs} | Train MSE Loss: {epoch_loss / len(loader):.2f}")

    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
    print(f"\n[+] Shared backbone saved to {OUTPUT_MODEL_PATH}")

    # Save held-out per-patient splits for the fine-tuning stage
    np.save("held_out_patient_splits.npy", held_out_test, allow_pickle=True)
    print(f"[+] Held-out per-patient train/test splits saved to held_out_patient_splits.npy")
    print(f"[+] {len(held_out_test)} patients available for fine-tuning.")


if __name__ == "__main__":
    run_pretraining()