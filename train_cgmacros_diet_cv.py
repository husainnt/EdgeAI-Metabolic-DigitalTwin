"""
train_cgmacros_diet_cv.py
==========================================================
Diet-track training and evaluation on the CGMacros window cache built by
build_cgmacros_window_cache.py. Leave-one-subject-out (LOSO) cross-validation,
because there are only about 9-13 usable subjects.

Per held-out subject, three arms are trained FROM THE SAME INITIALISATION under
one fixed modality configuration each (same method as the anchor runs):
    context   : calibration context only            [HR 0, Act 0, Sleep 0, SpO2 0, Diet 0, Ctx 1]
    ctx_wear  : HR + Activity + context (no diet)    [1, 1, 0, 0, 0, 1]
    full      : HR + Activity + Diet + context       [1, 1, 0, 0, 1, 1]
    ctx_diet  : Diet + context only (no HR/Activity) [0, 0, 0, 0, 1, 1]   (optional arm)
so the diet effect is the paired contrast  full - ctx_wear  (per held-out
subject, postprandial windows), and nothing else changes between the two arms.
ctx_diet - context isolates Diet from the HR/Activity encoders entirely.

Training baseline = MEAL-BLIND physics (mech_fore / mech_curr in the cache).
The meal-aware physics is read only to print a reference column.

Weight transfer: the AI-READI curriculum checkpoint initialises the fusion
Transformer, residual head, HR, Sleep, SpO2 and calibration-context encoders.
enc_diet is always freshly initialised. enc_activity is freshly initialised too
(--activity-init scratch, default): CGMacros gives METs and Calories (Activity),
not steps and walking fraction, so the transferred weights mean something else.

Normalisation: HR and calibration context use the checkpoint's AI-READI statistics
(so the transferred encoders see the scale they were trained on). Activity and Diet
statistics are fitted on the TRAINING subjects of each fold only, never on the
held-out subject. Diet features: macros -> log1p, time-since-meal clipped at
--tsm-clip-min and converted to hours, has_had_meal left as 0/1.

No epoch selection on the held-out subject. By default every arm trains for a fixed
--epochs. With --inner-val K, K TRAINING subjects are set aside inside each fold and
used only to pick the best epoch (early stopping, postprandial RMSE) for each arm,
the same rule for every arm; the held-out subject is never used for selection.
Results are saved per fold, so a stopped run resumes where it left off.

Usage (from D:\\FYP\\CODE):
    # code-path check (numbers are NOT meaningful)
    python train_cgmacros_diet_cv.py --smoke

    # the main run
    python train_cgmacros_diet_cv.py --run-name cv_main

    # same, with the three broken-baseline subjects kept out of TRAINING (still evaluated)
    python train_cgmacros_diet_cv.py --run-name cv_no_broken_train --exclude-train 39,47,49

    # nested early stopping plus the diet-only arm
    python train_cgmacros_diet_cv.py --run-name cv_es --exclude-train 39,47,49 --inner-val 2 --arms context,ctx_wear,full,ctx_diet
"""

import os
import sys
import glob
import json
import time
import hashlib
import argparse
import numpy as np

# Here I put the model code on the path the same way pretrain_six_encoder_population.py does
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "glycemic_twin", "ml_layer"))

# [HR, Activity, Sleep, SpO2, Diet, Calibration-context], same slot order as hybrid_twin.py
ARMS = {
    "context":  [0, 0, 0, 0, 0, 1],
    "ctx_wear": [1, 1, 0, 0, 0, 1],
    "full":     [1, 1, 0, 0, 1, 1],
    "ctx_diet": [0, 0, 0, 0, 1, 1],
}
POSTPRANDIAL_MIN = 180  # same 3 h window the gate and the builder use

# (label, arm_a, arm_b): the paired difference is RMSE(arm_a) - RMSE(arm_b); negative = arm_a better
CONTRASTS = [
    ("DIET effect        (full - ctx_wear)", "full", "ctx_wear"),
    ("wearable effect    (ctx_wear - context)", "ctx_wear", "context"),
    ("DIET-ONLY effect  (ctx_diet - context)", "ctx_diet", "context"),
    ("full vs context    (full - context)", "full", "context"),
    ("full vs physics    (full - phys_blind)", "full", "phys_blind"),
    ("ctx_wear vs physics (ctx_wear - phys_blind)", "ctx_wear", "phys_blind"),
]


# ----------------------------------------------------------------------------
# Data loading and normalisation (numpy only)
# ----------------------------------------------------------------------------
def load_cache(cache_dir, drop_subjects):
    """Stacks every subject's CLEAN windows (no pre-calibration, no anchor in the horizon)."""
    # Here I skip the builder's temp files so a run started mid-build never reads a half-written file
    files = sorted(f for f in glob.glob(os.path.join(cache_dir, "subject_*.npz")) if not f.endswith(".tmp.npz"))
    keys = ("hr", "act", "glc", "diet", "mech_curr", "mech_fore", "mech_fore_aware", "y",
            "tsm_target_min", "target_row")
    cols = {k: [] for k in keys + ("has", "pid")}
    empty = []
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            pid = int(d["patient_id"])
            if pid in drop_subjects:
                continue
            keep = ~(d["pre_calib"] | d["anchor_in_horizon"])
            n = int(keep.sum())
            if n == 0:
                empty.append(pid)
                continue
            for k in keys:
                cols[k].append(d[k][keep])
            cols["has"].append(np.tile(d["has"].astype(np.float32), (n, 1)))
            cols["pid"].append(np.full(n, pid, dtype=np.int64))
    if not cols["y"]:
        raise SystemExit(f"[!] No usable subject_*.npz in {cache_dir}. Run build_cgmacros_window_cache.py first.")
    return {k: np.concatenate(v) for k, v in cols.items()}, empty


def diet_features(diet, tsm_clip_min):
    out = diet.astype(np.float32).copy()
    out[:, :5] = np.log1p(np.clip(out[:, :5], 0, None))          # five macros, skewed -> log1p
    out[:, 5] = np.minimum(out[:, 5], tsm_clip_min) / 60.0        # minutes since last meal -> hours, clipped
    return out                                                    # column 6 (has_had_meal) stays 0/1


def fit_fold_stats(D, diet_t, train_mask, src_stats):
    def ms(a):
        return [float(a.mean()), max(float(a.std()), 1e-3)]
    st = {}
    if src_stats is not None:
        for k in ("hr", "glc_tsc_hours", "glc_calib_mgdl"):
            st[k] = [float(src_stats[k][0]), float(src_stats[k][1])]
    else:
        st["hr"] = ms(D["hr"][train_mask])
        st["glc_tsc_hours"] = ms(D["glc"][train_mask][:, 0])
        st["glc_calib_mgdl"] = ms(D["glc"][train_mask][:, 1])
    st["act_mets"] = ms(D["act"][train_mask][..., 0])
    st["act_cal"] = ms(D["act"][train_mask][..., 1])
    st["diet_mean"] = [float(x) for x in diet_t[train_mask][:, :6].mean(axis=0)]
    st["diet_std"] = [max(float(x), 1e-3) for x in diet_t[train_mask][:, :6].std(axis=0)]
    return st


def apply_norm(D, diet_t, st):
    has = D["has"]
    out = {}
    out["hr"] = ((D["hr"] - st["hr"][0]) / st["hr"][1]) * has[:, 0:1]
    a0 = (D["act"][..., 0] - st["act_mets"][0]) / st["act_mets"][1]
    a1 = (D["act"][..., 1] - st["act_cal"][0]) / st["act_cal"][1]
    out["act"] = np.stack([a0, a1], axis=-1) * has[:, 1][:, None, None]
    out["glc"] = np.stack([
        (D["glc"][:, 0] - st["glc_tsc_hours"][0]) / st["glc_tsc_hours"][1],
        (D["glc"][:, 1] - st["glc_calib_mgdl"][0]) / st["glc_calib_mgdl"][1],
    ], axis=-1)
    dz = diet_t.copy()
    dz[:, :6] = (dz[:, :6] - np.array(st["diet_mean"], dtype=np.float32)) / np.array(st["diet_std"], dtype=np.float32)
    out["diet"] = dz
    return {k: v.astype(np.float32) for k, v in out.items()}


# ----------------------------------------------------------------------------
# Statistics (numpy only)
# ----------------------------------------------------------------------------
def rmse(a, b):
    if len(a) == 0:
        return float("nan")
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return float(np.sqrt(np.mean(d ** 2)))


def paired_stats(diff):
    """diff = per-subject (arm_a RMSE - arm_b RMSE). Negative = arm_a better."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n < 2:
        return {"n": n, "mean": float("nan"), "se": float("nan"), "frac_improved": float("nan"),
                "verdict": "too few subjects"}
    mean = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(n))
    if mean + 2 * se < 0:
        verdict = "IMPROVES (mean below 0 by more than 2 SE)"
    elif mean - 2 * se > 0:
        verdict = "WORSE (mean above 0 by more than 2 SE)"
    else:
        verdict = "not distinguishable from zero"
    return {"n": n, "mean": mean, "se": se, "frac_improved": float(np.mean(diff < 0)), "verdict": verdict}


def summarize(folds, arms, calibrated_thresh):
    """folds: list of dicts with y, mech_fore, mech_fore_aware, tsm_target_min, pred_<arm>.
    Returns (per-subject rows, contrast results)."""
    rows = []
    for f in folds:
        pp = (f["tsm_target_min"] >= 0) & (f["tsm_target_min"] <= POSTPRANDIAL_MIN)
        row = {"subject": int(f["pid"]), "n_all": int(len(f["y"])), "n_pp": int(pp.sum())}
        series = {"phys_blind": f["mech_fore"], "phys_aware": f["mech_fore_aware"]}
        series.update({a: f[f"pred_{a}"] for a in arms})
        for name, p in series.items():
            row[f"{name}_all"] = rmse(p, f["y"])
            row[f"{name}_pp"] = rmse(p[pp], f["y"][pp])
        row["group"] = "calibrated" if row["phys_blind_all"] <= calibrated_thresh else "broken_baseline"
        rows.append(row)

    results = {}
    for subset in ("pp", "all"):
        for grp in ("all_subjects", "calibrated", "broken_baseline"):
            sel = [r for r in rows if grp == "all_subjects" or r["group"] == grp]
            for label, a, b in CONTRASTS:
                if (a != "phys_blind" and a not in arms) or (b != "phys_blind" and b not in arms):
                    continue
                diff = [r[f"{a}_{subset}"] - r[f"{b}_{subset}"] for r in sel]
                results[f"{subset}|{grp}|{label}"] = paired_stats(diff)
    return rows, results


# ----------------------------------------------------------------------------
# Model I/O (torch)
# ----------------------------------------------------------------------------
def build_model(device, bound):
    from hybrid_twin import HybridResidualTwin  # imported here so the numpy helpers stay importable without torch
    model = HybridResidualTwin(max_residual_bound=bound).to(device)
    dims = {
        "hr": model.enc_hr.lstm.input_size,
        "act": model.enc_activity.lstm.input_size,
        "diet": model.enc_diet.net[0].in_features,
        "glc": model.enc_calib_glucose.net[0].in_features,
    }
    # Here I fail loudly if the encoder shapes drift from what the cache provides
    assert dims["act"] == 2 and dims["diet"] == 7 and dims["glc"] == 2, \
        f"Encoder input sizes changed, cache layout no longer matches: {dims}"
    return model, dims


def init_model(device, args, ckpt, seed):
    import torch
    torch.manual_seed(seed)
    model, dims = build_model(device, args.residual_bound)
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state"])
        # Here I draw FRESH random weights for the encoders whose inputs mean something new
        # in CGMacros: Diet always, Activity unless --activity-init transfer
        fresh, _ = build_model(device, args.residual_bound)
        model.enc_diet.load_state_dict(fresh.enc_diet.state_dict())
        if args.activity_init == "scratch":
            model.enc_activity.load_state_dict(fresh.enc_activity.state_dict())
    return model, dims


def make_inputs(dims, T, idx, mask):
    """mask: (B,6) float [HR, Act, Sleep, SpO2, Diet, Context]. Availability flags gate HR and Activity."""
    import torch
    B = idx.numel()
    has = T["has"][idx]
    hr = T["hr"][idx].unsqueeze(-1)
    if dims["hr"] != 1:
        # Here I tile the single real HR channel to the encoder's input width, exactly like the AI-READI script
        hr = hr.repeat(1, 1, dims["hr"])
    zeros_seq = torch.zeros(B, hr.shape[1], 1, device=mask.device)
    zeros_b = torch.zeros(B, device=mask.device)
    return {
        "hr": {"data": hr, "mask": mask[:, 0] * has[:, 0]},
        "activity": {"data": T["act"][idx], "mask": mask[:, 1] * has[:, 1]},
        "sleep": {"data": zeros_seq, "mask": zeros_b},   # CGMacros has no sleep stream: always off
        "spo2": {"data": zeros_seq, "mask": zeros_b},    # CGMacros has no SpO2 stream: always off
        "diet": {"data": T["diet"][idx], "mask": mask[:, 4]},
        "glucose": {"data": T["glc"][idx], "mask": mask[:, 5]},
    }


def train_arm(model, dims, T, tr_idx, arm_mask, args, device, tag, val=None):
    import torch
    import torch.nn as nn
    new_params = list(model.enc_activity.parameters()) + list(model.enc_diet.parameters())
    new_ids = {id(p) for p in new_params}
    backbone = [p for p in model.parameters() if id(p) not in new_ids]
    # Here I fine-tune the transferred backbone gently and let the fresh encoders learn faster
    opt = torch.optim.AdamW([{"params": backbone, "lr": args.lr_backbone},
                             {"params": new_params, "lr": args.lr_new}], weight_decay=1e-4)
    crit = nn.MSELoss()
    pat = torch.tensor(arm_mask, dtype=torch.float32, device=device)
    last = float("nan")
    best_score, best_state, best_epoch = float("inf"), None, args.epochs
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), device=device)]
        if args.smoke:
            perm = perm[: 20 * args.batch_size]
        total = torch.zeros((), device=device)
        for i in range(0, len(perm), args.batch_size):
            b = perm[i:i + args.batch_size]
            mask = pat.unsqueeze(0).repeat(len(b), 1)
            y_pred, _ = model(make_inputs(dims, T, b, mask), T["mech_curr"][b], T["mech_fore"][b])
            loss = crit(y_pred, T["y"][b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.detach() * len(b)
        last = float(total) / len(perm)
        msg = ""
        if val is not None:
            # Here I score the INNER validation subjects (never the held-out subject) on their
            # postprandial windows, the same rule for every arm, and keep the best epoch
            pv = predict(model, dims, T, val["idx"], arm_mask, 4096, device)
            sel = val["pp"] if val["pp"].any() else np.ones(len(pv), dtype=bool)
            score = rmse(pv[sel], val["y"][sel])
            if score < best_score:
                best_score, best_epoch = score, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            msg = f" | inner-val pp RMSE {score:.2f}{' *' if best_epoch == epoch else ''}"
        if epoch == 1 or epoch == args.epochs or epoch % 5 == 0 or (val is not None and best_epoch == epoch):
            print(f"      [{tag}] epoch {epoch:02d}/{args.epochs} train MSE {last:.1f}{msg}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return last, best_epoch


def predict(model, dims, T, idx, pattern, batch_size, device):
    import torch
    model.eval()
    pat = torch.tensor(pattern, dtype=torch.float32, device=device)
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), batch_size):
            b = idx[i:i + batch_size]
            mask = pat.unsqueeze(0).repeat(len(b), 1)
            y_pred, _ = model(make_inputs(dims, T, b, mask), T["mech_curr"][b], T["mech_fore"][b])
            out.append(y_pred.squeeze(-1))
    return torch.cat(out).cpu().numpy()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_ids(s):
    return {int(x) for x in s.split(",") if x.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.path.join("results", "cgmacros_window_cache"))
    ap.add_argument("--out-dir", default=os.path.join("results", "cgmacros_diet_cv"))
    ap.add_argument("--run-name", default="cv_main")
    ap.add_argument("--init-ckpt", default=os.path.join("results", "six_encoder_population", "curriculum", "best_model.pt"),
                    help="AI-READI curriculum checkpoint, or 'none' to train everything from scratch")
    ap.add_argument("--arms", default="context,ctx_wear,full")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr-backbone", type=float, default=2e-4)
    ap.add_argument("--lr-new", type=float, default=1e-3)
    ap.add_argument("--activity-init", choices=["scratch", "transfer"], default="scratch")
    ap.add_argument("--residual-bound", type=float, default=80.0, help="residual head bound in mg/dL (default 80)")
    ap.add_argument("--tsm-clip-min", type=float, default=360.0)
    ap.add_argument("--calibrated-rmse", type=float, default=60.0,
                    help="subjects whose meal-blind physics RMSE (clean windows) is at or below this are 'calibrated'")
    ap.add_argument("--drop-subjects", default="", help="comma-separated subject IDs removed from everything")
    ap.add_argument("--exclude-train", default="", help="comma-separated subject IDs never used for TRAINING (still evaluated)")
    ap.add_argument("--inner-val", type=int, default=0,
                    help="number of TRAINING subjects held out inside each fold for early stopping (0 = fixed epochs)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="recompute folds that already have results")
    ap.add_argument("--smoke", action="store_true", help="2 folds, 2 epochs, 20 steps per epoch: code-path check only")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"[!] unknown arm '{a}'. Choose from {list(ARMS)}")
    drop, excl_train = parse_ids(args.drop_subjects), parse_ids(args.exclude_train)
    if args.smoke:
        args.epochs = 2

    # ---- data ----
    D, empty = load_cache(args.cache_dir, drop)
    subjects = sorted(int(p) for p in np.unique(D["pid"]))
    print(f"[+] Subjects with clean windows: {subjects} ({len(D['y'])} windows)")
    if empty:
        print(f"[!] Subjects in the cache with ZERO clean windows (skipped): {sorted(empty)}")
    if len(subjects) < 3:
        raise SystemExit("[!] Need at least 3 subjects for cross-validation.")

    # Here I print how often the meal-blind baseline error exceeds the residual head's bound, since
    # a saturated head cannot fully correct those windows no matter how good the Diet encoder is
    pp_all = (D["tsm_target_min"] >= 0) & (D["tsm_target_min"] <= POSTPRANDIAL_MIN)
    gap = np.abs(D["y"] - D["mech_fore"])
    print(f"[+] Share of postprandial windows with |y - physics| > {args.residual_bound:g} mg/dL: "
          f"{100 * float((gap[pp_all] > args.residual_bound).mean()):.1f}% "
          f"(all clean windows: {100 * float((gap > args.residual_bound).mean()):.1f}%)")

    diet_t = diet_features(D["diet"], args.tsm_clip_min)

    # ---- run folder and compatibility guard ----
    run_dir = os.path.join(args.out_dir, args.run_name + ("_smoke" if args.smoke else ""))
    os.makedirs(run_dir, exist_ok=True)
    keep_keys = ("arms", "epochs", "batch_size", "lr_backbone", "lr_new", "activity_init", "residual_bound",
                 "tsm_clip_min", "drop_subjects", "exclude_train", "seed", "init_ckpt", "inner_val")
    sig = {k: getattr(args, k) for k in keep_keys}
    # Here I record the subject list too: adding a subject changes every fold's training set,
    # so old fold results must not be silently reused
    sig["subjects"] = subjects
    sig_path = os.path.join(run_dir, "args.json")
    if os.path.exists(sig_path) and not args.force:
        with open(sig_path) as f:
            old = json.load(f)
        if old != sig:
            raise SystemExit(f"[!] {sig_path} was made with different settings:\n  old {old}\n  new {sig}\n"
                             "Use a new --run-name, or --force to recompute every fold.")
    with open(sig_path, "w") as f:
        json.dump(sig, f, indent=2)

    # ---- torch and checkpoint ----
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | run dir: {run_dir}")
    ckpt, src_stats = None, None
    if args.init_ckpt.lower() != "none":
        if not os.path.exists(args.init_ckpt):
            raise SystemExit(f"[!] Checkpoint not found: {args.init_ckpt}\n"
                             "    Copy the curriculum run's best_model.pt there, or pass --init-ckpt none.")
        ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        src_stats = ckpt["stats"]
        print(f"[+] Initialising from {args.init_ckpt} (train_config = {ckpt.get('train_config')})")
        if ckpt.get("train_config") != "curriculum":
            print("[!] This checkpoint was not trained with the curriculum config; expected 'curriculum'.")

    # ---- LOSO folds ----
    fold_subjects = subjects[:2] if args.smoke else subjects
    t_start = time.time()
    for fold, pid in enumerate(fold_subjects):
        fold_path = os.path.join(run_dir, f"fold_{pid}.npz")
        if os.path.exists(fold_path) and not args.force:
            print(f"[i] fold {pid}: already done, skipping")
            continue
        te_np = np.where(D["pid"] == pid)[0]
        eligible = [p for p in subjects if p != pid and p not in excl_train]
        # Here I pick the inner-validation subjects by a hash of (subject, fold), so the choice is
        # deterministic, differs between folds, and never touches the held-out subject
        inner = sorted(eligible, key=lambda p: hashlib.md5(f"{p}|{pid}".encode()).hexdigest())[:args.inner_val]
        if args.inner_val and len(eligible) - len(inner) < 2:
            raise SystemExit(f"[!] fold {pid}: --inner-val {args.inner_val} leaves too few training subjects")
        tr_np = np.where(np.isin(D["pid"], [p for p in eligible if p not in inner]))[0]
        va_np = np.where(np.isin(D["pid"], inner))[0]
        if len(tr_np) == 0:
            raise SystemExit(f"[!] fold {pid}: no training windows left (check --exclude-train)")
        extra = f" + inner-val {sorted(inner)}" if inner else ""
        print(f"\n=== fold {fold + 1}/{len(fold_subjects)}: hold out subject {pid} "
              f"({len(te_np)} windows) | train on {len(np.unique(D['pid'][tr_np]))} subjects, {len(tr_np)} windows{extra} ===", flush=True)

        train_mask = np.zeros(len(D["y"]), dtype=bool)
        train_mask[tr_np] = True
        stats = fit_fold_stats(D, diet_t, train_mask, src_stats)
        Xn = apply_norm(D, diet_t, stats)
        T = {k: torch.from_numpy(v).to(device) for k, v in Xn.items()}
        T["has"] = torch.from_numpy(D["has"]).to(device)
        T["mech_curr"] = torch.from_numpy(D["mech_curr"]).unsqueeze(-1).to(device)
        T["mech_fore"] = torch.from_numpy(D["mech_fore"]).unsqueeze(-1).to(device)
        T["y"] = torch.from_numpy(D["y"]).unsqueeze(-1).to(device)
        tr_idx = torch.from_numpy(tr_np).long().to(device)
        te_idx = torch.from_numpy(te_np).long().to(device)
        val = None
        if len(va_np):
            val = {"idx": torch.from_numpy(va_np).long().to(device), "y": D["y"][va_np], "pp": pp_all[va_np]}

        save = {"pid": np.array(pid), "y": D["y"][te_np], "mech_fore": D["mech_fore"][te_np],
                "mech_fore_aware": D["mech_fore_aware"][te_np], "tsm_target_min": D["tsm_target_min"][te_np],
                "target_row": D["target_row"][te_np]}
        pp = (save["tsm_target_min"] >= 0) & (save["tsm_target_min"] <= POSTPRANDIAL_MIN)
        for a_i, arm in enumerate(arms):
            t0 = time.time()
            model, dims = init_model(device, args, ckpt, seed=args.seed + 1000 * fold + a_i)
            last, best_ep = train_arm(model, dims, T, tr_idx, ARMS[arm], args, device, f"{pid}/{arm}", val)
            pred = predict(model, dims, T, te_idx, ARMS[arm], 4096, device)
            save[f"pred_{arm}"] = pred
            save[f"best_epoch_{arm}"] = np.array(best_ep)
            # Here I print the held-out RMSE for monitoring only; it never selects an epoch or a setting
            print(f"    {arm:<9} held-out RMSE all {rmse(pred, save['y']):6.2f} | postprandial "
                  f"{rmse(pred[pp], save['y'][pp]):6.2f} | physics {rmse(save['mech_fore'], save['y']):6.2f} "
                  f"| best epoch {best_ep} | {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(fold_path, **save)
        el = (time.time() - t_start) / 60
        print(f"  [saved] {fold_path} | elapsed {el:.1f} min", flush=True)

    # ---- aggregate every finished fold ----
    folds = []
    for pid in fold_subjects:
        p = os.path.join(run_dir, f"fold_{pid}.npz")
        if os.path.exists(p):
            with np.load(p, allow_pickle=False) as d:
                folds.append({k: d[k] for k in d.files})
    rows, results = summarize(folds, arms, args.calibrated_rmse)

    print("\n" + "=" * 100)
    print(f"PER-SUBJECT RMSE (mg/dL), held-out subject, clean windows | pp = within {POSTPRANDIAL_MIN} min after a meal")
    print("=" * 100)
    head = f"{'subj':>5} {'group':<16} {'n_pp':>5} " + " ".join(f"{h:>10}" for h in
            ["phys_blind", "phys_aware"] + arms) + "   | postprandial only:"
    print(head)
    for r in rows:
        line = f"{r['subject']:>5} {r['group']:<16} {r['n_pp']:>5} " + " ".join(
            f"{r[f'{n}_all']:>10.2f}" for n in ["phys_blind", "phys_aware"] + arms)
        line += "   | " + " ".join(f"{r[f'{n}_pp']:>8.2f}" for n in ["phys_blind", "phys_aware"] + arms)
        print(line)

    print("\n" + "=" * 100)
    print("PAIRED CONTRASTS over held-out subjects (mean RMSE difference in mg/dL, negative = first arm better)")
    print("=" * 100)
    for subset, sname in (("pp", "POSTPRANDIAL windows"), ("all", "ALL clean windows")):
        print(f"\n--- {sname} ---")
        for grp in ("all_subjects", "calibrated", "broken_baseline"):
            for label, a, b in CONTRASTS:
                s = results.get(f"{subset}|{grp}|{label}")
                if s is None or s["n"] == 0:
                    continue
                print(f"  [{grp:<15}] {label:<44}: {s['mean']:+7.2f} (SE {s['se']:.2f}, n={s['n']}, "
                      f"{100 * s['frac_improved']:.0f}% improved) -- {s['verdict']}")

    csv_path = os.path.join(run_dir, "per_subject_cv.csv")
    if rows:
        cols = list(rows[0].keys())
        with open(csv_path, "w") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(str(round(r[c], 4)) if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({"args": sig, "subjects": [r["subject"] for r in rows], "contrasts": results}, f, indent=2)
    print(f"\n[SAVED] {run_dir}: per_subject_cv.csv, results.json, fold_<id>.npz (out-of-fold predictions)")


if __name__ == "__main__":
    main()