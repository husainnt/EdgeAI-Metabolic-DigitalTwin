# ==========================================================
# Does glucose follow a time-of-day pattern?
# Average glucose by hour of day
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
file = r"D:\FYP\DATA_SET\cgmacros-a-scientific-dataset-for-personalized-nutrition-and-diet-monitoring-1.0.0\CGMacros_dateshifted365\CGMacros\CGMacros-001\CGMacros-001.csv"

df = pd.read_csv(file)

# Convert columns
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Libre GL"] = pd.to_numeric(df["Libre GL"], errors="coerce")

# Remove missing glucose values
df = df.dropna(subset=["Libre GL"])

# Extract hour of day
df["Hour"] = df["Timestamp"].dt.hour

# Average glucose for each hour
hourly = df.groupby("Hour")["Libre GL"].mean()

print(hourly)

# Plot
plt.figure(figsize=(10,5))

plt.plot(
    hourly.index,
    hourly.values,
    marker='o',
    linewidth=2
)

plt.xticks(range(24))
plt.xlabel("Hour of Day")
plt.ylabel("Average Glucose (mg/dL)")
plt.title("Average Glucose by Hour of Day")
plt.grid(True)

plt.show()