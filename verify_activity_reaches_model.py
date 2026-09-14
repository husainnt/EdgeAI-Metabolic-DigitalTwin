"""
Ground-truth check: pulls one real batch from the dataloader and inspects
act_window / has_act directly, rather than trusting any printed banner.

Run this any time you want to be certain a modality is actually flowing
into training, not just present as a column somewhere upstream.

Usage:
    python verify_activity_reaches_model.py
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "glycemic_twin", "ml_layer")))
from window_generator import get_dataloader


def verify():
    train_loader, _ = get_dataloader("results/cohort/patient_1031_merged.csv", batch_size=32, train_days=8)

    # Here I pull exactly one real batch, the same object build_inputs_dict
    # would receive during actual training -- no shortcuts, no re-derivation.
    batch = next(iter(train_loader))

    act_window = batch["act_window"]
    has_act = batch["has_act"]

    print(f"act_window shape: {tuple(act_window.shape)}  (expect [batch, 12, 2])")
    print(f"has_act values in this batch (should be all 1.0 if Activity column is present): "
          f"{has_act.flatten().tolist()[:8]}...")

    steps_channel = act_window[:, :, 0]
    walk_channel = act_window[:, :, 1]

    print(f"\nsteps_sum channel -- mean: {steps_channel.mean():.2f}, "
          f"max: {steps_channel.max():.2f}, nonzero fraction: {(steps_channel != 0).float().mean():.3f}")
    print(f"walking_frac channel -- mean: {walk_channel.mean():.4f}, "
          f"max: {walk_channel.max():.4f}, nonzero fraction: {(walk_channel != 0).float().mean():.3f}")

    if steps_channel.abs().sum().item() == 0:
        print("\n[!] CRITICAL: act_window's steps channel is ALL ZERO in this batch.")
        print("    Activity is NOT actually reaching the model despite has_act -- investigate.")
    else:
        print("\n[+] Confirmed: real, non-zero Activity data is present in the actual training tensor.")


if __name__ == "__main__":
    verify()