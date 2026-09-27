"""
export_twin_visualization_data.py
==========================================================
Exports a real, already-trained-and-validated patient run as a per-window
JSON file Zunaira's Unity scene can play back frame-by-frame, so her organ
color/animation logic can be built and tested against real numbers right
now -- independent of whether the Jetson/BLE pipeline exists yet.

Reuses the project's OWN validated code paths rather than reimplementing
anything:
    - window_generator.get_dataloader() -- same train/val split, same
      physics engine, as every other result in this project
    - hybrid_twin.HybridResidualTwin -- the actual trained six-encoder model
    - train_patient1031.build_inputs_dict / get_expected_input_size /
      adapt_seq_dim -- the exact same forward-pass wiring already used to
      produce every reported hybrid RMSE number, imported not duplicated

Exports the VAL split (unseen-during-training data) by default -- this is
an honest choice, not cherry-picked training data, and it's a naturally
demo-sized ~2-day segment for Patient 1031.

WHAT IS REAL vs WHAT IS A CLEARLY-FLAGGED PLACEHOLDER, per window:
    glucose_real            REAL -- true CGM reading
    glucose_physics         REAL -- validated simglucose warm-start forecast
    glucose_hybrid          REAL -- the actual trained model's prediction
    heart_rate              REAL -- last real HR value in the input window
    pancreas_secretion_proxy REAL -- T2DPancreaticController.basal_for()
                             evaluated on the real glucose reading, using
                             the project's own locked T2D parameters (same
                             controller class imported from
                             run_selective_warmstart_simglucose_hybrid.py)
    liver_output_proxy       NOT YET REAL -- always null. No clean
                             single-function hepatic-output proxy exists
                             yet without deeper ODE-state integration work.
                             Exported as null on purpose so it's obviously
                             not real data, not silently zero.
    degradation_level         DEMO-ONLY, NOT a real inference output --
                             cycles through levels 1-5 across the export
                             purely so Zunaira has something to test her
                             degradation-ladder visuals against. This is
                             NOT what a deployed device would report; the
                             real ladder depends on live sensor
                             availability, which this historical AI-READI
                             CSV doesn't represent. Flagged in "meta" too.
    meal_event_active         Always 0 for AI-READI patients -- AI-READI
                             has no diet data (see project dataset notes).
                             Will become real once the CGMacros diet track
                             is wired into the six-encoder model.

Usage:
    python export_twin_visualization_data.py --patient 1031
    python export_twin_visualization_data.py --patient 1031 --checkpoint results/hybrid_residual_twin_patient1031.pt
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "glycemic_twin", "ml_layer"))

from window_generator import get_dataloader
from hybrid_twin import HybridResidualTwin
from train_patient1031 import build_inputs_dict
from run_selective_warmstart_simglucose_hybrid import build_calibrated_t2d_patient, T2DPancreaticController

OUT_DIR = "results/twin_visualization_export"


def build_secretion_controller():
    """Here I reuse the project's own locked T2D parameters and controller
    class directly, rather than re-deriving them, so the secretion proxy
    is computed with the exact same physiology as every other result."""
    patient = build_calibrated_t2d_patient()
    params = patient._params
    basal_rate = params["u2ss"] * params["BW"] / 6000
    return T2DPancreaticController(gb=params["Gb"], basal_rate=basal_rate), basal_rate, params["Gb"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-path", default="results/cohort/patient_1031_merged.csv")
    ap.add_argument("--checkpoint", default="results/hybrid_residual_twin_patient1031.pt")
    ap.add_argument("--patient", default="1031", help="label used in the output filename")
    ap.add_argument("--split", choices=["val", "train"], default="val",
                    help="val = unseen-during-training data (recommended for an honest demo)")
    args = ap.parse_args()

    device = torch.device("cpu")
    print(f"Loading model from {args.checkpoint} ...")
    model = HybridResidualTwin().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    print(f"Loading data from {args.csv_path} (using the {args.split} split) ...")
    train_loader, val_loader = get_dataloader(args.csv_path, batch_size=32, train_days=8)
    loader = val_loader if args.split == "val" else train_loader

    controller, basal_rate, gb = build_secretion_controller()

    frames = []
    level_cycle = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]  # demo-only cycling, see docstring
    with torch.no_grad():
        for batch in loader:
            b_size = batch["cgm_window"].size(0)
            g_mech_curr = batch["g_mech_curr"].to(device)
            g_mech_fore = batch["g_mech_fore"].to(device)
            y_target = batch["y_target"].to(device)
            hr_window = batch["hr_window"]  # (B, 12, 1), last real HR in the window

            # Here I evaluate with all modalities on (mask_tensor=None -> all-ones
            # in build_inputs_dict), the SAME convention train_patient1031.py uses
            # for its own reported validation numbers
            inputs = build_inputs_dict(model, batch, device, mask_tensor=None)
            y_pred, _ = model(inputs, g_mech_curr, g_mech_fore)

            for i in range(b_size):
                g_real = float(y_target[i].item())
                secretion = float(controller.basal_for(g_real))
                frames.append({
                    "glucose_real": round(g_real, 2),
                    "glucose_physics": round(float(g_mech_fore[i].item()), 2),
                    "glucose_hybrid": round(float(y_pred[i].item()), 2),
                    "heart_rate": round(float(hr_window[i, -1, 0].item()), 1),
                    "pancreas_secretion_proxy": round(secretion, 5),
                    "liver_output_proxy": None,  # not yet real, see docstring
                    "degradation_level": level_cycle[len(frames) % len(level_cycle)],  # demo-only
                    "meal_event_active": 0,  # AI-READI has no diet data
                })

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"patient_{args.patient}_{args.split}_demo.json")

    payload = {
        "meta": {
            "patient": args.patient,
            "split": args.split,
            "n_frames": len(frames),
            "frame_interval_minutes": 5,
            "field_notes": {
                "glucose_real": "true CGM reading, mg/dL",
                "glucose_physics": "validated simglucose warm-start forecast, mg/dL",
                "glucose_hybrid": "trained six-encoder model prediction, mg/dL",
                "heart_rate": "real HR at window end, bpm",
                "pancreas_secretion_proxy": (
                    f"REAL, computed from T2DPancreaticController.basal_for(real_glucose). "
                    f"Range: {basal_rate:.5f} (basal, at Gb={gb:.1f} mg/dL) up to the "
                    f"controller's max_secretion cap. Higher = more insulin secretion."
                ),
                "liver_output_proxy": "NOT YET REAL -- always null, do not visualize yet",
                "degradation_level": (
                    "DEMO-ONLY, cycles 1-5 for testing your ladder visuals -- NOT a real "
                    "inference output. Do not present this field's values as measured."
                ),
                "meal_event_active": "always 0 for this AI-READI patient (no diet data source)",
            },
        },
        "frames": frames,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[SAVED] {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()