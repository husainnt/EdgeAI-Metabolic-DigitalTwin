"""
Derives the real T2D subject list from bio.csv using standard ADA clinical
diagnostic criteria -- there's no explicit diagnosis column in this
dataset, so this is the correct, citable way to identify T2D participants
rather than guessing a label:

    A1c >= 6.5%              -> diabetes
    Fasting glucose >= 126   -> diabetes
    (either condition alone is sufficient per ADA criteria)

Cross-checks the resulting count against this project's own prior note of
"14 confirmed T2D" out of 45 total CGMacros participants -- if the numbers
don't match, that's worth investigating before trusting either figure.

Usage:
    python derive_cgmacros_t2d_subset.py
"""

import os
import pandas as pd

CGMACROS_ROOT = r"D:\FYP\DATA_SET\cgmacros-a-scientific-dataset-for-personalized-nutrition-and-diet-monitoring-1.0.0\CGMacros_dateshifted365\CGMacros"
A1C_THRESHOLD = 6.5
FASTING_GLU_THRESHOLD = 126

OUTPUT_CSV = "results/cgmacros_t2d_subject_list.csv"


def derive_t2d_subset():
    bio_path = os.path.join(CGMACROS_ROOT, "bio.csv")
    df = pd.read_csv(bio_path)

    # Here I strip whitespace from column names, since a couple of them
    # (e.g. 'Body weight ') had trailing spaces in the raw header
    df.columns = [c.strip() for c in df.columns]

    a1c_col = "A1c PDL (Lab)"
    fasting_col = "Fasting GLU - PDL (Lab)"

    if a1c_col not in df.columns or fasting_col not in df.columns:
        print(f"[!] Expected columns not found. Real columns: {df.columns.tolist()}")
        return

    # Here I use ONLY A1c, matching the published CGMacros paper's exact
    # stated criteria (Das et al., Scientific Data 2025): "15 had no
    # pre-existing diabetes (HbA1c<5.7%), 16 had pre-diabetes
    # (5.7<=HbA1c<=6.4%), and 14 had type 2 diabetes (HbA1c>6.4%)".
    # NOTE: this is narrower than general ADA criteria (which would also
    # diagnose from fasting glucose alone) -- deliberately matching the
    # dataset's own published methodology so our subject counts match
    # theirs exactly, not a broader clinical definition.
    df["is_t2d"] = df[a1c_col] > 6.4
    df["is_prediabetic"] = (df[a1c_col] >= 5.7) & (df[a1c_col] <= 6.4)
    df["is_healthy"] = df[a1c_col] < 5.7

    t2d_subjects = df[df["is_t2d"]]["subject"].tolist()
    prediabetic_subjects = df[df["is_prediabetic"]]["subject"].tolist()
    healthy_subjects = df[df["is_healthy"]]["subject"].tolist()

    print(f"[i] Total participants in bio.csv: {len(df)}")
    print(f"[+] T2D (A1c>={A1C_THRESHOLD} or fasting GLU>={FASTING_GLU_THRESHOLD}): {len(t2d_subjects)}")
    print(f"    Subject IDs: {t2d_subjects}")
    print(f"[i] Prediabetic: {len(prediabetic_subjects)}")
    print(f"[i] Healthy: {len(healthy_subjects)}")

    expected_t2d_count = 14
    if len(t2d_subjects) != expected_t2d_count:
        print(f"\n[!] MISMATCH: derived {len(t2d_subjects)} T2D subjects, but this project's")
        print(f"    prior notes say 14. Worth double-checking the criteria/thresholds above")
        print(f"    before trusting either number -- don't silently pick one.")
    else:
        print(f"\n[+] Matches the expected count of 14 -- good cross-check.")

    os.makedirs("results", exist_ok=True)
    df[["subject", a1c_col, fasting_col, "is_t2d", "is_prediabetic", "is_healthy"]].to_csv(OUTPUT_CSV, index=False)
    print(f"\n[SAVED] {OUTPUT_CSV}")


if __name__ == "__main__":
    derive_t2d_subset()