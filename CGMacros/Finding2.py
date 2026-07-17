# ==========================================================
# Does glucose correlate with activity?
# Compare glucose during sedentary and active periods
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
file = r"D:\FYP\DATA_SET\cgmacros-a-scientific-dataset-for-personalized-nutrition-and-diet-monitoring-1.0.0\CGMacros_dateshifted365\CGMacros\CGMacros-001\CGMacros-001.csv"

df = pd.read_csv(file)

# Convert columns to correct types
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["METs"] = pd.to_numeric(df["METs"], errors="coerce")
df["Libre GL"] = pd.to_numeric(df["Libre GL"], errors="coerce")

# Remove rows with missing values
df = df.dropna(subset=["METs", "Libre GL"])

# Define activity levels
active = df[df["METs"] >= 2]
sedentary = df[df["METs"] < 2]

print("Total Samples:", len(df))
print("Active Samples:", len(active))
print("Sedentary Samples:", len(sedentary))

print("\nAverage Glucose")
print("---------------------------")
print("Active:", round(active["Libre GL"].mean(), 2), "mg/dL")
print("Sedentary:", round(sedentary["Libre GL"].mean(), 2), "mg/dL")

# Plot comparison
plt.figure(figsize=(8,6))

plt.boxplot(
    [sedentary["Libre GL"], active["Libre GL"]],
    tick_labels=["Sedentary", "Active"]
)

plt.ylabel("Glucose (mg/dL)")
plt.title("Glucose During Active vs Sedentary Periods")
plt.grid(True)

plt.show()