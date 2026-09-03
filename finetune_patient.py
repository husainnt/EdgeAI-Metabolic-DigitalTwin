"""
Per-Patient Fine-Tuning
========================
Loads the shared HR+Sleep backbone (pretrained on the pooled cohort) and
fine-tunes ONLY the small fusion_head (residual head) per patient, using
that patient's own calibration-window training split. The HR and Sleep
LSTM encoders + context arm stay FROZEN -- this is the concrete answer to
the panel's personalization objection: "one shared model pretrained once
on population data, then only a small residual head fine-tuned per
patient during calibration."

Reports RMSE/MAE for:
    1. Physics-only baseline (simglucose, no ML at all)
    2. Shared backbone with NO fine-tuning (zero-shot on this patient)
    3. Shared backbone + fine-tuned residual head (the real FYP-II result)

Compare result #3 against the old per-patient-independent-training numbers
(Patient 1031: 33.61 mg/dL, Patient 1205: 13.38 mg/dL) to see whether
shared pretraining + light fine-tuning matches, beats, or falls short of
training from scratch per patient -- report this honestly either way.
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import numpy as np

from pretrain_shared_backbone import SharedHRSleepResidualLSTM

MODEL_PATH = "shared_pretrained_backbone.pt"
NORM_STATS_PATH = "shared_pretrain_norm_stats.npz"
HELD_OUT_SPLITS_PATH = "held_out_patient_splits.npy"

# Kept in sync with pretrain_shared_backbone.py's EXCLUDE_PATIENT_IDS --
# 1027 is a known severe outlier (physics RMSE 283.45 mg/dL vs ~30-80 for
# everyone else), independently reproducing the same flag already set in
# find_patient_2_candidates.py's EXCLUDE_IDS.
EXCLUDE_PATIENT_IDS = {"1027"}

torch.manual_seed(42)
np.random.seed(42)


class FineTuneDataset(Dataset):
    def __init__(self, X_hr, X_sleep, X_ctx, y):
        self.X_hr = torch.tensor(X_hr, dtype=torch.float32)
        self.X_sleep = torch.tensor(X_sleep, dtype=torch.float32)
        self.X_ctx = torch.tensor(X_ctx, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_hr[idx], self.X_sleep[idx], self.X_ctx[idx], self.y[idx]


def normalize(X_hr, X_sleep, tsc, calib_val, stats):
    X_hr_n = (X_hr - stats["hr_mean"]) / stats["hr_std"]
    X_sleep_n = (X_sleep - stats["sleep_mean"]) / stats["sleep_std"]
    tsc_n = (tsc - stats["tsc_mean"]) / stats["tsc_std"]
    calib_n = (calib_val - stats["calib_mean"]) / stats["calib_std"]
    X_ctx_n = np.stack([tsc_n, calib_n], axis=1)
    return X_hr_n, X_sleep_n, X_ctx_n


def finetune_one_patient(pid, patient_data, stats, finetune_epochs=30):
    X_hr_tr_full, X_sleep_tr_full, X_ctx_tr_full = normalize(
        patient_data["X_hr_train"], patient_data["X_sleep_train"],
        patient_data["tsc_train"], patient_data["calib_val_train"], stats
    )
    y_res_tr_full = patient_data["y_real_train"] - patient_data["y_sim_train"]

    # Further split this patient's OWN training portion into sub-train/val
    # (chronological, 85/15) purely for early stopping -- the held-out test
    # split is never touched here. This is what was missing before: without
    # an internal validation signal, fine-tuning had no way to detect
    # overfitting and just used whatever the final epoch happened to produce.
    n_tr = len(y_res_tr_full)
    val_split = int(n_tr * 0.85)
    if val_split < 10 or (n_tr - val_split) < 5:
        # Too little data to carve out a meaningful validation set --
        # fall back to using the full training set with no early stopping.
        val_split = n_tr

    X_hr_tr, X_sleep_tr, X_ctx_tr = X_hr_tr_full[:val_split], X_sleep_tr_full[:val_split], X_ctx_tr_full[:val_split]
    y_res_tr = y_res_tr_full[:val_split]
    X_hr_val, X_sleep_val, X_ctx_val = X_hr_tr_full[val_split:], X_sleep_tr_full[val_split:], X_ctx_tr_full[val_split:]
    y_res_val = y_res_tr_full[val_split:]
    has_val = len(y_res_val) > 0

    X_hr_te, X_sleep_te, X_ctx_te = normalize(
        patient_data["X_hr"], patient_data["X_sleep"],
        patient_data["tsc"], patient_data["calib_val"], stats
    )
    y_real_te = patient_data["y_real"]
    y_sim_te = patient_data["y_sim"]

    rmse_physics = np.sqrt(np.mean((y_real_te - y_sim_te) ** 2))

    # --- Load shared backbone, freeze encoders ---
    model = SharedHRSleepResidualLSTM(hidden_dim=32, context_dim=2)
    model.load_state_dict(torch.load(MODEL_PATH))

    for param in model.hr_encoder.parameters():
        param.requires_grad = False
    for param in model.sleep_encoder.parameters():
        param.requires_grad = False
    for param in model.fc_context.parameters():
        param.requires_grad = False
    # Only fusion_head remains trainable

    # --- Zero-shot eval (before fine-tuning) ---
    model.eval()
    with torch.no_grad():
        t_hr_te = torch.tensor(X_hr_te, dtype=torch.float32)
        t_sleep_te = torch.tensor(X_sleep_te, dtype=torch.float32)
        t_ctx_te = torch.tensor(X_ctx_te, dtype=torch.float32)
        zero_shot_res = model(t_hr_te, t_sleep_te, t_ctx_te).numpy()
    rmse_zero_shot = np.sqrt(np.mean((y_real_te - (y_sim_te + zero_shot_res)) ** 2))

    # --- Fine-tune fusion_head only, with validation-based checkpointing ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=finetune_epochs, eta_min=1e-5)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        FineTuneDataset(X_hr_tr, X_sleep_tr, X_ctx_tr, y_res_tr),
        batch_size=16, shuffle=True
    )

    if has_val:
        t_hr_val = torch.tensor(X_hr_val, dtype=torch.float32)
        t_sleep_val = torch.tensor(X_sleep_val, dtype=torch.float32)
        t_ctx_val = torch.tensor(X_ctx_val, dtype=torch.float32)
        t_y_val = torch.tensor(y_res_val, dtype=torch.float32)

    best_val_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.fusion_head.state_dict().items()}

    for epoch in range(finetune_epochs):
        model.train()
        for b_hr, b_sleep, b_ctx, b_y in loader:
            optimizer.zero_grad()
            pred = model(b_hr, b_sleep, b_ctx)
            loss = loss_fn(pred, b_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if has_val:
            model.eval()
            with torch.no_grad():
                val_pred = model(t_hr_val, t_sleep_val, t_ctx_val)
                val_loss = loss_fn(val_pred, t_y_val).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.fusion_head.state_dict().items()}

    # Restore the best-validation-loss checkpoint (not just the final epoch)
    if has_val:
        model.fusion_head.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        finetuned_res = model(t_hr_te, t_sleep_te, t_ctx_te).numpy()
    rmse_finetuned = np.sqrt(np.mean((y_real_te - (y_sim_te + finetuned_res)) ** 2))
    mae_finetuned = np.mean(np.abs(y_real_te - (y_sim_te + finetuned_res)))

    return {
        "patient_id": pid,
        "rmse_physics": round(float(rmse_physics), 2),
        "rmse_zero_shot": round(float(rmse_zero_shot), 2),
        "rmse_finetuned": round(float(rmse_finetuned), 2),
        "mae_finetuned": round(float(mae_finetuned), 2),
        "gain_over_physics_pct": round(float((rmse_physics - rmse_finetuned) / rmse_physics * 100), 2),
        "used_early_stopping": has_val,
    }


def run_finetune_evaluation(patient_ids=None):
    stats_npz = np.load(NORM_STATS_PATH)
    stats = {k: float(stats_npz[k]) for k in stats_npz.files}

    held_out = np.load(HELD_OUT_SPLITS_PATH, allow_pickle=True).item()

    if patient_ids is None:
        patient_ids = [pid for pid in held_out.keys() if pid not in EXCLUDE_PATIENT_IDS]

    results = []
    for pid in patient_ids:
        if pid not in held_out:
            print(f"[!] Patient {pid} not in held-out set, skipping.")
            continue
        result = finetune_one_patient(pid, held_out[pid], stats)
        results.append(result)
        print(f"Patient {result['patient_id']:>6} | Physics RMSE: {result['rmse_physics']:6.2f} | "
              f"Zero-shot RMSE: {result['rmse_zero_shot']:6.2f} | "
              f"Fine-tuned RMSE: {result['rmse_finetuned']:6.2f} | "
              f"Gain: {result['gain_over_physics_pct']:+.1f}% | "
              f"EarlyStop: {'Y' if result['used_early_stopping'] else 'N (too little data)'}")

    if results:
        avg_gain = np.mean([r["gain_over_physics_pct"] for r in results])
        print(f"\n[+] Average gain over physics across {len(results)} patients: {avg_gain:+.1f}%")

    return results


if __name__ == "__main__":
    # Default: evaluate on ALL patients in the current held-out set (whatever
    # was used in the pretraining run -- e.g. the 20-patient smoke test).
    # Patients 1031/1205 are NOT guaranteed to be in this cohort (they were
    # selected via different criteria than the oral-med T2D filter) -- check
    # with `dir results\cohort\patient_1031_merged.csv` before assuming they're
    # available, and pass patient_ids=["1031","1205"] explicitly once confirmed.
    run_finetune_evaluation(patient_ids=None)