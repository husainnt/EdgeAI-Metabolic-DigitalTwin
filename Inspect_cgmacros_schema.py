"""
Inspects the real CGMacros schema before writing any diet-encoder
extraction code -- same "verify against real files first" discipline used
for every other dataset in this project.

Checks:
    1. Full column list of one participant's per-timestep CSV (not the
       truncated view Excel shows)
    2. A sample of rows where Meal Type is actually populated (to see real
       macro values, not just the mostly-empty in-between rows)
    3. bio.xlsx's columns, to find whatever field identifies T2D status

Usage:
    python inspect_cgmacros_schema.py
"""

import os
import pandas as pd

CGMACROS_ROOT = r"D:\FYP\DATA_SET\cgmacros-a-scientific-dataset-for-personalized-nutrition-and-diet-monitoring-1.0.0\CGMacros_dateshifted365\CGMacros"
SAMPLE_PARTICIPANT = "CGMacros-001"


def inspect_participant_csv():
    csv_path = os.path.join(CGMACROS_ROOT, SAMPLE_PARTICIPANT, f"{SAMPLE_PARTICIPANT}.csv")
    if not os.path.exists(csv_path):
        # Here I try .xlsx as a fallback, since the screenshot showed an Excel icon
        alt_path = os.path.join(CGMACROS_ROOT, SAMPLE_PARTICIPANT, f"{SAMPLE_PARTICIPANT}.xlsx")
        if os.path.exists(alt_path):
            df = pd.read_excel(alt_path)
            print(f"[i] Loaded {alt_path}")
        else:
            print(f"[!] Neither {csv_path} nor {alt_path} found -- check the exact file extension/path.")
            return
    else:
        df = pd.read_csv(csv_path)
        print(f"[i] Loaded {csv_path}")

    print(f"\n=== FULL COLUMN LIST ({len(df.columns)} columns) ===")
    print(df.columns.tolist())

    print(f"\n=== First 3 rows (all columns) ===")
    print(df.head(3).to_string())

    if "Meal Type" in df.columns:
        meal_rows = df[df["Meal Type"].notna()]
        print(f"\n=== Rows where Meal Type is populated: {len(meal_rows)} / {len(df)} total rows ===")
        print(meal_rows.head(10).to_string())
    else:
        print("\n[!] No 'Meal Type' column found -- check the real column names above for the actual name used.")


def inspect_bio_file():
    bio_path = os.path.join(CGMACROS_ROOT, "bio.csv")
    if not os.path.exists(bio_path):
        print(f"\n[!] {bio_path} not found.")
        return

    df_bio = pd.read_csv(bio_path)
    print(f"\n=== bio.csv columns ({len(df_bio)} participants) ===")
    print(df_bio.columns.tolist())
    print("\nFirst 5 rows:")
    print(df_bio.head(5).to_string())
    print("\nAll rows (so we can see every participant's values, given there are only ~45):")
    print(df_bio.to_string())


if __name__ == "__main__":
    inspect_participant_csv()
    inspect_bio_file()