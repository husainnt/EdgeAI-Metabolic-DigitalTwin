import os
import pandas as pd

# Absolute dataset directory configurations
dataset_path = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset"
participants_path = os.path.join(dataset_path, "participants.tsv")
manifest_path = os.path.join(dataset_path, "wearable_blood_glucose", "manifest.tsv")

print("=== SEARCHING FOR TYPICAL/MODERATE MULTIMODAL PATIENTS ===")

# Load master participant files and glucose manifests
df_part = pd.read_csv(participants_path, sep="\t")
df_man = pd.read_csv(manifest_path, sep="\t")

# Filter target cohort for perfect multimodal tracking completeness
target_group = "oral_medication_and_or_non_insulin_injectable_medication_controlled"
multimodal_cohort = df_part[
    (df_part["study_group"] == target_group) &
    (df_part["wearable_blood_glucose"] == True) &
    (df_part["wearable_activity_monitor"] == True)
]

# Merge demographics with manifest metrics using inner join
candidates = pd.merge(
    multimodal_cohort[["person_id", "age"]],
    df_man[["person_id", "average_glucose_level_mg_dl", "glucose_level_record_count", "glucose_sensor_sampling_duration_days"]],
    on="person_id"
)

# Apply Claude's clinical window filter (140 to 190 mg/dL)
moderate_candidates = candidates[
    (candidates["average_glucose_level_mg_dl"] >= 140.0) &
    (candidates["average_glucose_level_mg_dl"] <= 190.0)
].sort_values(by="glucose_level_record_count", ascending=False).reset_index(drop=True)

print(f"\nFound {len(moderate_candidates)} candidates matching moderate-control window & multimodal criteria.")

if not moderate_candidates.empty:
    print("\n=== Top 10 Best-Fit Moderate Canonical Candidates ===")
    print(moderate_candidates.head(10).to_string(index=False))
else:
    print("\n[WARNING] No patients matched the narrow clinical criteria window.")