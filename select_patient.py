import os
import pandas as pd

# path to the local dataset directory
dataset_path = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset"
participants_file = os.path.join(dataset_path, "participants.tsv")

# load dataset
df = pd.read_csv(participants_file, sep="\t")

print("=== Unique Study Groups ===")
print(df["study_group"].value_counts())

print("\n=== Top T2D Cohort Candidates with CGM Data ===")
# filtering for participants who have wearable blood glucose tracking enabled
cgm_enabled = df[df["wearable_blood_glucose"] == True]

# print the top candidate options to select our canonical twin
print(cgm_enabled[["person_id", "study_group", "age"]].head(15).to_string(index=False))