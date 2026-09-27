"""
pretrain_six_encoder_population.py
==========================================================
Trains the SIX-ENCODER HybridResidualTwin (Transformer fusion) across the
whole cohort using the window cache built by build_six_encoder_window_cache.py,
then evaluates it on patients it has NEVER seen, under fixed modality masks.
No simglucose is needed here: the physics forecasts are already in the cache,
so this runs on Colab with only glycemic_twin/ml_layer/*.py and the cache.

What is different from the earlier-chat version of this script:
  1. PATIENT-LEVEL split (about 70/15/15 train/val/test by patient, decided by
     a hash of the patient ID so it stays stable as the cache grows). The
     model never sees a test patient. No random window split, so no leakage
     from overlapping neighbouring windows.
  2. The best epoch is chosen on the VAL patients; results are reported on the
     TEST patients only.
  3. Inputs are standardized with statistics from TRAIN patients only (the
     encoders do no normalization of their own). log1p is applied to steps.
  4. Pre-calibration windows (baseline = flat glucose[0], not simglucose) are
     excluded by default, so "physics" always means simglucose output.
  5. Training masks = your 5-level degradation curriculum MIXED with
     independent per-modality dropout (calibration context on), so every
     configuration the ablation evaluates is one the model trained on.
  6. Data lives on the GPU as tensors: no DataLoader, so the T4 is not starved.
  7. Paired per-patient statistics for each modality's contribution, so a
     0.3 mg/dL difference is not mistaken for a real effect.

Usage:
    # quick code-path check on whatever is cached (numbers are NOT meaningful)
    python pretrain_six_encoder_population.py --smoke

    # the main run: one shared model, curriculum masks, 6-config ablation
    python pretrain_six_encoder_population.py --run-name curriculum

    # anchor runs: retrain from scratch under ONE fixed configuration, the same
    # method as the single-patient context-only controls
    python pretrain_six_encoder_population.py --train-config context --run-name anchor_context
    python pretrain_six_encoder_population.py --train-config all --run-name anchor_all
"""

import os
import sys
import glob
import json
import time
import hashlib
import argparse
import numpy as np
import torch
import torch.nn as nn

# Here I put the model code on the path the same way the earlier scripts did
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "glycemic_twin", "ml_layer"))

# [HR, Activity, Sleep, SpO2, Diet, Calibration-context] -- same key order as
# sample_dynamic_mask in train_patient1031.py
CONFIGS = {
    "Context-only":    [0, 0, 0, 0, 0, 1],
    "+ HR only":       [1, 0, 0, 0, 0, 1],
    "+ Sleep only":    [0, 0, 1, 0, 0, 1],
    "+ SpO2 only":     [0, 0, 0, 1, 0, 1],
    "+ Activity only": [0, 1, 0, 0, 0, 1],
    "+ ALL real":      [1, 1, 1, 1, 0, 1],  # diet excluded: no real diet data in AI-READI
}
CONFIG_ALIASES = {
    "context": "Context-only", "hr": "+ HR only", "sleep": "+ Sleep only",
    "spo2": "+ SpO2 only", "activity": "+ Activity only", "all": "+ ALL real",
}
# Which per-patient availability flag a configuration needs (index into has = [hr, act, sleep, spo2])
CONFIG_REQUIRES = {"+ HR only": 0, "+ Activity only": 1, "+ Sleep only": 2, "+ SpO2 only": 3}

# Here I keep the five degradation levels from train_patient1031.py exactly
LEVEL_MASKS = [
    [1, 1, 1, 1, 1, 1],  # level 1: everything on
    [1, 1, 0, 0, 0, 0],  # level 2: HR + Activity
    [1, 0, 0, 0, 0, 0],  # level 3: HR only
    [0, 0, 0, 0, 1, 1],  # level 4: events only (diet is forced off later, so context only)
    [0, 0, 0, 0, 0, 0],  # level 5: pure ODE fallback
]


# ----------------------------------------------------------------------------
# Data loading (numpy only)
# ----------------------------------------------------------------------------
def load_cache(cache_dir, max_patients=0, keep_precalib=False):
    """Stacks every cached patient's train + later windows into flat arrays."""
    # Here I skip the builder's temp files (patient_<id>.npz.tmp.npz) so a run started while the
    # cache is still being built never tries to read a half-written file
    files = sorted(f for f in glob.glob(os.path.join(cache_dir, "patient_*.npz")) if not f.endswith(".tmp.npz"))
    if max_patients:
        files = files[:max_patients]
    if not files:
        raise SystemExit(f"[!] No patient_*.npz files found in {cache_dir}. Run build_six_encoder_window_cache.py first.")

    keys = ("hr", "act", "sleep", "spo2", "glc", "mech_curr", "mech_fore", "y")
    cols = {k: [] for k in keys + ("has", "pid")}
    n_precalib_dropped = 0
    for f in files:
        with np.load(f) as d:
            pid = int(d["patient_id"])
            for split in ("train", "later"):
                n = len(d[f"{split}_y"])
                if n == 0:
                    continue
                keep = np.ones(n, dtype=bool) if keep_precalib else ~d[f"{split}_pre_calib"]
                n_keep = int(keep.sum())
                n_precalib_dropped += n - n_keep
                if n_keep == 0:
                    continue
                for k in keys:
                    cols[k].append(d[f"{split}_{k}"][keep])
                cols["has"].append(np.tile(d[f"{split}_has"].astype(np.float32), (n_keep, 1)))
                cols["pid"].append(np.full(n_keep, pid, dtype=np.int64))
    D = {k: np.concatenate(v) for k, v in cols.items()}
    return D, n_precalib_dropped, len(files)


def bucket(pid):
    # Here I hash the patient ID so the split stays the same when more patients get cached
    return int(hashlib.md5(str(pid).encode()).hexdigest(), 16) % 100


def assign_splits(pid_array):
    """Returns integer array: 0 = train, 1 = val, 2 = test (per window)."""
    up, inv = np.unique(pid_array, return_inverse=True)
    b = np.array([bucket(p) for p in up])
    split_u = np.where(b < 70, 0, np.where(b < 85, 1, 2))
    return split_u[inv], up, split_u


# ----------------------------------------------------------------------------
# Normalization (statistics from TRAIN patients only)
# ----------------------------------------------------------------------------
def steps_log1p(act):
    out = act.copy()
    out[..., 0] = np.log1p(np.clip(out[..., 0], 0, None))
    return out


def fit_norm_stats(D, act_t, train_mask):
    def ms(a):
        return [float(a.mean()), max(float(a.std()), 1e-3)]

    has = D["has"]
    st = {}
    for name, key, k in (("hr", "hr", 0), ("sleep", "sleep", 2), ("spo2", "spo2", 3)):
        sel = train_mask & (has[:, k] == 1)
        st[name] = ms(D[key][sel]) if sel.any() else [0.0, 1.0]
    sel = train_mask & (has[:, 1] == 1)
    st["act_steps_log1p"] = ms(act_t[sel][..., 0]) if sel.any() else [0.0, 1.0]
    st["act_walk_frac"] = ms(act_t[sel][..., 1]) if sel.any() else [0.0, 1.0]
    st["glc_tsc_hours"] = ms(D["glc"][train_mask][:, 0])
    st["glc_calib_mgdl"] = ms(D["glc"][train_mask][:, 1])
    return st


def apply_norm(D, act_t, st):
    has = D["has"]
    out = {}
    out["hr"] = ((D["hr"] - st["hr"][0]) / st["hr"][1]) * has[:, 0:1]
    out["sleep"] = ((D["sleep"] - st["sleep"][0]) / st["sleep"][1]) * has[:, 2:3]
    out["spo2"] = ((D["spo2"] - st["spo2"][0]) / st["spo2"][1]) * has[:, 3:4]
    a0 = (act_t[..., 0] - st["act_steps_log1p"][0]) / st["act_steps_log1p"][1]
    a1 = (act_t[..., 1] - st["act_walk_frac"][0]) / st["act_walk_frac"][1]
    out["act"] = np.stack([a0, a1], axis=-1) * has[:, 1][:, None, None]
    out["glc"] = np.stack([
        (D["glc"][:, 0] - st["glc_tsc_hours"][0]) / st["glc_tsc_hours"][1],
        (D["glc"][:, 1] - st["glc_calib_mgdl"][0]) / st["glc_calib_mgdl"][1],
    ], axis=-1)
    return {k: v.astype(np.float32) for k, v in out.items()}


# ----------------------------------------------------------------------------
# Per-patient statistics (numpy only)
# ----------------------------------------------------------------------------
def per_patient_rmse(pred, y, pid):
    up, inv = np.unique(pid, return_inverse=True)
    se = (pred.astype(np.float64) - y.astype(np.float64)) ** 2
    mse = np.bincount(inv, weights=se) / np.bincount(inv)
    return up, np.sqrt(mse)


def per_patient_has(has, pid):
    up, inv = np.unique(pid, return_inverse=True)
    ph = np.zeros((len(up), has.shape[1]), dtype=np.float32)
    np.maximum.at(ph, inv, has)
    return up, ph


def paired_stats(diff):
    """diff = per-patient (config RMSE - context-only RMSE). Negative = better."""
    n = len(diff)
    if n < 2:
        return {"n": n, "mean": float("nan"), "se": float("nan"), "frac_improved": float("nan"), "verdict": "too few patients"}
    mean = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(n))
    if mean + 2 * se < 0:
        verdict = "IMPROVES (mean below 0 by more than 2 SE)"
    elif mean - 2 * se > 0:
        verdict = "WORSE than context alone (mean above 0 by more than 2 SE)"
    else:
        verdict = "not distinguishable from zero"
    return {"n": n, "mean": mean, "se": se, "frac_improved": float(np.mean(diff < 0)), "verdict": verdict}


# ----------------------------------------------------------------------------
# Model I/O (torch)
# ----------------------------------------------------------------------------
def build_model(device):
    from hybrid_twin import HybridResidualTwin  # imported here so numpy helpers stay importable without the model code
    model = HybridResidualTwin().to(device)
    # Here I read the real encoder input sizes so a wrong shape fails loudly instead of silently tiling
    dims = {
        "hr": model.enc_hr.lstm.input_size,
        "act": model.enc_activity.lstm.input_size,
        "sleep": model.enc_sleep.lstm.input_size,
        "spo2": model.enc_spo2.lstm.input_size,
        "diet": model.enc_diet.net[0].in_features,
        "glc": model.enc_calib_glucose.net[0].in_features,
    }
    assert dims["act"] == 2 and dims["sleep"] == 1 and dims["spo2"] == 1 and dims["glc"] == 2, \
        f"Encoder input sizes changed, cache layout no longer matches: {dims}"
    return model, dims


def make_inputs(dims, T, idx, mask):
    """mask: (B,6) float [HR, Act, Sleep, SpO2, Diet, Context]. Availability flags gate it."""
    B = idx.numel()
    has = T["has"][idx]
    hr = T["hr"][idx].unsqueeze(-1)
    if dims["hr"] != 1:
        # Here I tile the single real HR channel to the encoder's 4 inputs, exactly like adapt_seq_dim
        # did in train_patient1031.py. This is a known limitation, not extra real information.
        hr = hr.repeat(1, 1, dims["hr"])
    zeros_b = torch.zeros(B, device=mask.device)
    return {
        "hr": {"data": hr, "mask": mask[:, 0] * has[:, 0]},
        "activity": {"data": T["act"][idx], "mask": mask[:, 1] * has[:, 1]},
        "sleep": {"data": T["sleep"][idx].unsqueeze(-1), "mask": mask[:, 2] * has[:, 2]},
        "spo2": {"data": T["spo2"][idx].unsqueeze(-1), "mask": mask[:, 3] * has[:, 3]},
        "diet": {"data": torch.zeros(B, dims["diet"], device=mask.device), "mask": zeros_b},  # no diet data: always off
        "glucose": {"data": T["glc"][idx], "mask": mask[:, 5]},
    }


def sample_train_masks(B, p_levels, device):
    lv = torch.randint(0, len(LEVEL_MASKS), (B,), device=device)
    m_lv = torch.tensor(LEVEL_MASKS, dtype=torch.float32, device=device)[lv]
    m_ind = torch.zeros(B, 6, device=device)
    m_ind[:, :4] = (torch.rand(B, 4, device=device) < 0.5).float()  # independent dropout per wearable
    m_ind[:, 5] = 1.0                                               # calibration context on
    use_lv = (torch.rand(B, 1, device=device) < p_levels).float()
    return use_lv * m_lv + (1 - use_lv) * m_ind


@torch.no_grad()
def predict(model, dims, T, idx, pattern, batch_size, device):
    model.eval()
    pat = torch.tensor(pattern, dtype=torch.float32, device=device)
    out = []
    for i in range(0, len(idx), batch_size):
        b = idx[i:i + batch_size]
        mask = pat.unsqueeze(0).repeat(len(b), 1)
        y_pred, _ = model(make_inputs(dims, T, b, mask), T["mech_curr"][b], T["mech_fore"][b])
        out.append(y_pred.squeeze(-1))
    return torch.cat(out).cpu().numpy()


def rmse_mae(pred, y):
    d = pred.astype(np.float64) - y.astype(np.float64)
    return float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d)))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="results/six_encoder_cache")
    ap.add_argument("--out-dir", default="results/six_encoder_population")
    ap.add_argument("--run-name", default="curriculum")
    ap.add_argument("--train-config", default="curriculum",
                    choices=["curriculum"] + list(CONFIG_ALIASES.keys()),
                    help="'curriculum' = mixed masks; otherwise retrain under ONE fixed configuration")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--p-levels", type=float, default=0.5, help="share of windows using the 5-level curriculum (rest: independent dropout)")
    ap.add_argument("--max-patients", type=int, default=0)
    ap.add_argument("--keep-precalib", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="2 epochs of 20 steps; if a split is empty, reuse all windows (code-path test only)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke:
        args.epochs = 2
    run_dir = os.path.join(args.out_dir, args.run_name + ("_smoke" if args.smoke else ""))
    os.makedirs(run_dir, exist_ok=True)
    print(f"Device: {device} | run dir: {run_dir}")

    # ---- data ----
    D, n_pre, n_files = load_cache(args.cache_dir, args.max_patients, args.keep_precalib)
    N = len(D["y"])
    print(f"[+] Loaded {n_files} patient files, {N} windows "
          f"({'kept' if args.keep_precalib else 'dropped'} {n_pre} pre-calibration windows)")

    split_w, up, split_u = assign_splits(D["pid"])
    tr_np, va_np, te_np = (np.where(split_w == s)[0] for s in (0, 1, 2))
    print(f"[+] Patients: train {int((split_u == 0).sum())} | val {int((split_u == 1).sum())} | test {int((split_u == 2).sum())}")
    print(f"[+] Windows:  train {len(tr_np)} | val {len(va_np)} | test {len(te_np)}")
    if min(len(tr_np), len(va_np), len(te_np)) == 0:
        if args.smoke:
            print("[!] SMOKE MODE: a split is empty, reusing all windows for every split. Numbers are meaningless.")
            tr_np = va_np = te_np = np.arange(N)
        else:
            raise SystemExit("[!] A split is empty (too few patients cached). Use --smoke for a code-path check, or cache more patients.")

    act_t = steps_log1p(D["act"])
    train_mask = np.zeros(N, dtype=bool)
    train_mask[tr_np] = True
    stats = fit_norm_stats(D, act_t, train_mask)
    with open(os.path.join(run_dir, "norm_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    Xn = apply_norm(D, act_t, stats)
    print("[+] Normalization stats (train patients only):", {k: [round(x, 3) for x in v] for k, v in stats.items()})
    print("[+] Modality availability among train windows (share with real data): "
          f"HR {D['has'][tr_np, 0].mean():.2f}, Activity {D['has'][tr_np, 1].mean():.2f}, "
          f"Sleep {D['has'][tr_np, 2].mean():.2f}, SpO2 {D['has'][tr_np, 3].mean():.2f}")

    T = {k: torch.from_numpy(v).to(device) for k, v in Xn.items()}
    T["has"] = torch.from_numpy(D["has"]).to(device)
    T["mech_curr"] = torch.from_numpy(D["mech_curr"]).unsqueeze(-1).to(device)
    T["mech_fore"] = torch.from_numpy(D["mech_fore"]).unsqueeze(-1).to(device)
    T["y"] = torch.from_numpy(D["y"]).unsqueeze(-1).to(device)
    tr_idx, va_idx, te_idx = (torch.from_numpy(a).long().to(device) for a in (tr_np, va_np, te_np))
    y_np = D["y"]

    # ---- model ----
    model, dims = build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    fixed_name = None if args.train_config == "curriculum" else CONFIG_ALIASES[args.train_config]
    eval_configs = CONFIGS if fixed_name is None else {fixed_name: CONFIGS[fixed_name]}

    start_epoch, best_score, best_state = 1, float("inf"), None
    ckpt_path = os.path.join(run_dir, "inprogress.pt")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        if ck["n_windows"] != N:
            raise SystemExit(f"[!] {ckpt_path} was made on a different cache ({ck['n_windows']} windows vs {N} now). "
                             "Delete the run folder to start fresh.")
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        start_epoch, best_score, best_state = ck["epoch"] + 1, ck["best_score"], ck["best_state"]
        print(f"[i] Resuming from epoch {start_epoch}")

    print(f"\n[+] Training ({'curriculum masks' if fixed_name is None else 'fixed config: ' + fixed_name}), "
          f"{len(tr_idx)} windows, {args.epochs} epochs, batch {args.batch_size}")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), device=device)]
        if args.smoke:
            # Here I cap the smoke run at 20 steps per epoch so it finishes in about a minute even on a busy CPU
            perm = perm[: 20 * args.batch_size]
        total = torch.zeros((), device=device)
        for i in range(0, len(perm), args.batch_size):
            b = perm[i:i + args.batch_size]
            if fixed_name is None:
                mask = sample_train_masks(len(b), args.p_levels, device)
            else:
                mask = torch.tensor(CONFIGS[fixed_name], dtype=torch.float32, device=device).unsqueeze(0).repeat(len(b), 1)
            y_pred, _ = model(make_inputs(dims, T, b, mask), T["mech_curr"][b], T["mech_fore"][b])
            loss = criterion(y_pred, T["y"][b])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(b)
        train_mse = float(total) / len(perm)

        # Validation on VAL patients. Best epoch = lowest mean RMSE over the evaluated configurations.
        val_rmse = {}
        for name, pattern in eval_configs.items():
            pred = predict(model, dims, T, va_idx, pattern, 4096, device)
            val_rmse[name], _ = rmse_mae(pred, y_np[va_np])
        score = float(np.mean(list(val_rmse.values())))
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        shown = ", ".join(f"{n.replace('+ ', '').replace(' only', '')} {v:.2f}" for n, v in val_rmse.items())
        print(f"Epoch {epoch:03d}/{args.epochs} | train MSE {train_mse:.1f} | val RMSE [{shown}] | score {score:.2f}"
              f"{' *' if best_state is not None and score == best_score else ''} | {time.time() - t0:.0f}s", flush=True)

        torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "epoch": epoch,
                    "best_score": best_score, "best_state": best_state, "n_windows": N}, ckpt_path)

    # ---- test on unseen patients ----
    model.load_state_dict(best_state)
    torch.save({"model_state": best_state, "stats": stats, "train_config": args.train_config, "args": vars(args)},
               os.path.join(run_dir, "best_model.pt"))

    pid_te = D["pid"][te_np]
    y_te = y_np[te_np]
    phys_te = D["mech_fore"][te_np]
    phys_rmse, phys_mae = rmse_mae(phys_te, y_te)
    up_te, phys_pp = per_patient_rmse(phys_te, y_te, pid_te)
    _, has_pp = per_patient_has(D["has"][te_np], pid_te)

    print("\n" + "=" * 86)
    print(f"TEST on {len(up_te)} UNSEEN patients, {len(te_np)} windows (post-calibration windows only: {not args.keep_precalib})")
    print("=" * 86)
    print(f"{'Configuration':<18} | {'RMSE':>7} | {'MAE':>7} | {'Physics RMSE':>12} | {'vs Physics':>10} | {'patients better than physics':>28}")
    print("-" * 100)
    results, pp_rmse = {}, {}
    for name, pattern in eval_configs.items():
        pred = predict(model, dims, T, te_idx, pattern, 4096, device)
        rmse, mae = rmse_mae(pred, y_te)
        _, pp = per_patient_rmse(pred, y_te, pid_te)
        pp_rmse[name] = pp
        results[name] = {"rmse": rmse, "mae": mae, "physics_rmse": phys_rmse,
                         "pct_vs_physics": (phys_rmse - rmse) / phys_rmse * 100,
                         "frac_patients_better_than_physics": float(np.mean(pp < phys_pp))}
        print(f"{name:<18} | {rmse:>7.2f} | {mae:>7.2f} | {phys_rmse:>12.2f} | {results[name]['pct_vs_physics']:>+9.1f}% | "
              f"{results[name]['frac_patients_better_than_physics'] * 100:>27.0f}%")

    contributions = {}
    if fixed_name is None:
        print("\n" + "=" * 86)
        print("ISOLATED CONTRIBUTION vs Context-only, paired over test patients (RMSE difference, negative = better)")
        print("Only patients that actually HAVE the modality are included for each row.")
        print("=" * 86)
        for name in ("+ HR only", "+ Sleep only", "+ SpO2 only", "+ Activity only", "+ ALL real"):
            need = CONFIG_REQUIRES.get(name)
            eligible = np.ones(len(up_te), dtype=bool) if need is None else has_pp[:, need] == 1
            diff = pp_rmse[name][eligible] - pp_rmse["Context-only"][eligible]
            s = paired_stats(diff)
            contributions[name] = s
            print(f"  {name:<16}: {s['mean']:+.2f} mg/dL (SE {s['se']:.2f}, n={s['n']} patients, "
                  f"{s['frac_improved'] * 100:.0f}% of patients improved) -- {s['verdict']}")

    pd_rows = ["patient_id,physics_rmse," + ",".join(f'"{n}"' for n in pp_rmse) + ",has_hr,has_act,has_sleep,has_spo2"]
    for i, p in enumerate(up_te):
        pd_rows.append(f"{p},{phys_pp[i]:.3f}," + ",".join(f"{pp_rmse[n][i]:.3f}" for n in pp_rmse) + "," + ",".join(str(int(x)) for x in has_pp[i]))
    with open(os.path.join(run_dir, "per_patient_test.csv"), "w") as f:
        f.write("\n".join(pd_rows))
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({"args": vars(args), "n_test_patients": int(len(up_te)), "n_test_windows": int(len(te_np)),
                   "results": results, "contributions": contributions}, f, indent=2)
    print(f"\n[SAVED] {run_dir}: best_model.pt, norm_stats.json, results.json, per_patient_test.csv")


if __name__ == "__main__":
    main()