import os
import pandas as pd

# here i define the exact local path to your dataset from the folder screenshots
dataset_path = r"F:\FYP\aireadi_data\aireadi-data\d0665d3d-1439-4627-b1c0-e0f2cbed8ebc\dataset"
participants_file = os.path.join(dataset_path, "participants.tsv")

# here i load the tsv data array into a pandas dataframe
df = pd.read_csv(participants_file, sep="\t")

print("=== AI-READI PARTICIPANTS TSV INSPECTION ===")
print(f"Total participants found: {len(df)}")
print("\n=== Available Columns ===")
for col in df.columns:
    print(f"- {col}")

print("\n=== First 3 Participant Records ===")
print(df.head(3))