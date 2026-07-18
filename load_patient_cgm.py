import os
import pandas as pd

# here i define the local path to your bids dataset directory
dataset_path = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset"
participants_file = os.path.join(dataset_path, "participants.tsv")
cgm_dir = os.path.join(dataset_path, "wearable_blood_glucose")

# here i load the master list
df = pd.read_csv(participants_file, sep="\t")

print("=== MULTIMODAL COMPLETENESS CHECK ===")
target_group = "oral_medication_and_or_non_insulin_injectable_medication_controlled"

# here i verify participant 1004's data availability flags
p1004 = df[df["person_id"] == 1004]
if not p1004.empty:
    print("\n--- Candidate Patient 1004 Flags ---")
    print(p1004[["person_id", "age", "wearable_blood_glucose", "wearable_activity_monitor", "clinical_data"]].to_string(index=False))

# here i isolate patients in our exact cohort who possess both glucose and activity traces
multimodal_candidates = df[
    (df["study_group"] == target_group) &
    (df["wearable_blood_glucose"] == True) &
    (df["wearable_activity_monitor"] == True)
]

print(f"\nTotal perfect multimodal patients in oral-med cohort: {len(multimodal_candidates)}")

if not multimodal_candidates.empty:
    print("\nTop 5 Perfect Multimodal Candidates:")
    print(multimodal_candidates[["person_id", "age", "wearable_blood_glucose", "wearable_activity_monitor"]].head(5).to_string(index=False))
    
    # here i choose 1004 if fully complete, otherwise fall back to the first 100% complete patient
    has_1004_activity = not p1004.empty and p1004["wearable_activity_monitor"].values[0] == True
    chosen_id = 1004 if has_1004_activity else multimodal_candidates["person_id"].values[0]
    
    print(f"\n🔍 Finalizing Case-Study Selection -> Targeted Patient ID: {chosen_id}")
    
    # here i discover the file names for the selected patient
    if os.path.exists(cgm_dir):
        all_files = os.listdir(cgm_dir)
        matches = [f for f in all_files if f"sub-{chosen_id}" in f or str(chosen_id) in f]
        print(f"Found files/folders in glucose directory: {matches}")
    else:
        print(f"[ERROR] Could not find glucose directory path at: {cgm_dir}")
else:
    print("\n[WARNING] No patients found matching both glucose and activity tracker parameters.")