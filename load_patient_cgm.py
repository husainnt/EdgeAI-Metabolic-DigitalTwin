import os
import pandas as pd

# here i define the paths based on our directory discovery
base_cgm_dir = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset\wearable_blood_glucose"
manifest_path = os.path.join(base_cgm_dir, "manifest.tsv")
sub_dir = os.path.join(base_cgm_dir, "continuous_glucose_monitoring")

print("=== STEP 1: PARSING MANIFEST TSV ===")
if os.path.exists(manifest_path):
    manifest_df = pd.read_csv(manifest_path, sep="\t")
    print(f"Manifest loaded successfully. Total index rows: {len(manifest_df)}")
    print("\nAvailable Manifest Columns:")
    print(manifest_df.columns.tolist())
    print("\nFirst 5 Rows of Manifest:")
    print(manifest_df.head(5).to_string(index=False))
else:
    print(f"[WARNING] manifest.tsv not found at: {manifest_path}")

print("\n" + "="*50 + "\n")

print("=== STEP 2: INNER RAW DIRECTORY LISTING (first 20 items) ===")
if os.path.exists(sub_dir):
    sub_items = os.listdir(sub_dir)
    print(f"Total files/folders found inside sub-directory: {len(sub_items)}")
    for item in sub_items[:20]:
        print(f"- {item}")
else:
    print(f"[ERROR] Could not find inner path: {sub_dir}")