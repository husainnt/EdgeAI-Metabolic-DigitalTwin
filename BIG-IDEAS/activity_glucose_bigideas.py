import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Files
# -----------------------------
acc_file = r"D:\FYP\DATA_SET\big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.3\001\ACC_001.csv"

dex_file = r"D:\FYP\DATA_SET\big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.3\001\Dexcom_001.csv"

# -----------------------------
# Load ACC
# -----------------------------
acc = pd.read_csv(acc_file)

acc.columns = acc.columns.str.strip()

acc["datetime"] = pd.to_datetime(acc["datetime"])

acc["Activity"] = np.sqrt(
    acc["acc_x"]**2 +
    acc["acc_y"]**2 +
    acc["acc_z"]**2
)

activity = (
    acc
    .set_index("datetime")
    .resample("5min")
    .mean()
)

# -----------------------------
# Load Dexcom
# -----------------------------
dex = pd.read_csv(dex_file)

dex = dex[dex["Event Type"]=="EGV"]

dex["Timestamp"] = pd.to_datetime(
    dex["Timestamp (YYYY-MM-DDThh:mm:ss)"]
)

dex["Glucose"] = dex["Glucose Value (mg/dL)"]

dex = dex.set_index("Timestamp")

# -----------------------------
# Merge
# -----------------------------
merged = dex.join(
    activity["Activity"],
    how="inner"
)

threshold = merged["Activity"].median()

active = merged[
    merged["Activity"] >= threshold
]

sedentary = merged[
    merged["Activity"] < threshold
]

plt.figure(figsize=(7,5))

plt.boxplot(
    [
        sedentary["Glucose"],
        active["Glucose"]
    ],
    labels=[
        "Sedentary",
        "Active"
    ]
)

plt.ylabel("Glucose (mg/dL)")
plt.title("BIG IDEAS: Activity vs Glucose")

plt.grid(True)

plt.show()

print()

print("Average sedentary glucose:",
      sedentary["Glucose"].mean())

print("Average active glucose:",
      active["Glucose"].mean())