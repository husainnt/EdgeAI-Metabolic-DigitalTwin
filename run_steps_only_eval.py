"""
Steps-Only Sparse Multimodal Evaluation for Patient 1031.
Uses real CGM data with mechanistic sparse calibration (2x/day),
with Garmin 'steps' as the single continuous auxiliary input.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Path to our unified multimodal dataset
DATASET_PATH = "results/patient_1031_real_cgm_hr_steps.csv"

def load_data_steps_only(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
        
    df = pd.read_csv(filepath)
    print(f"[+] Loaded dataset with {len(df)} rows.")
    
    # Feature extraction: CGM and Steps ONLY
    glucose = df['glucose_mg_dl'].values
    steps = df['steps'].values
    
    return df, glucose, steps


class StepsResidualLSTM(nn.Module):
    """
    LSTM designed to learn residual glucose forecast errors using 
    past CGM trajectory + continuous Step counts.
    """
    def __init__(self, input_dim=2, hidden_dim=32, num_layers=1):
        super(StepsResidualLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def evaluate_steps_only():
    df, glucose, steps = load_data_steps_only(DATASET_PATH)
    
    print("\n--- Running Standalone Steps-Only Benchmark ---")
    print(f"Total time steps: {len(glucose)}")
    print(f"Active step windows (>0 steps): {(steps > 0).sum()} / {len(steps)} ({((steps > 0).sum()/len(steps))*100:.1f}%)")
    
    # Here your sparse calibration + sSimGlucose hybrid mechanics run:
    # 1. Compute mechanistic baseline error (Sparse 2x/day warmstarts)
    # 2. Train StepsResidualLSTM on (CGM_window, Steps_window) -> Residual_Error
    # 3. Predict corrected trajectory
    
    # Placeholder for running your evaluation script:
    # Run the exact same pipeline as HR-only, passing steps as input_dim=2 feature vector


if __name__ == "__main__":
    evaluate_steps_only()