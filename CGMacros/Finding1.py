# Does glucose rise after meal events
# For this finding the implementation is as follows:every meal (rows where Carbs is recorded), extracts glucose from 2 hours before to 2 hours after, and overlays all meal responses.

import pandas as pd
import matplotlib.pyplot as plt

file = r"D:\FYP\DATA_SET\cgmacros-a-scientific-dataset-for-personalized-nutrition-and-diet-monitoring-1.0.0\CGMacros_dateshifted365\CGMacros\CGMacros-001\CGMacros-001.csv"

df = pd.read_csv(file)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Use Libre glucose
glucose = "Libre GL"

# Meal rows
meals = df[df["Carbs"].notna()]

plt.figure(figsize=(12,6))

for _, meal in meals.iterrows():

    t = meal["Timestamp"]

    window = df[
        (df["Timestamp"] >= t - pd.Timedelta(hours=2)) &
        (df["Timestamp"] <= t + pd.Timedelta(hours=2))
    ].copy()

    # minutes relative to meal
    window["Minutes"] = (
        window["Timestamp"] - t
    ).dt.total_seconds()/60

    plt.plot(window["Minutes"], window[glucose], alpha=0.4)

plt.axvline(0,color='red',linestyle='--',label='Meal')
plt.xlabel("Minutes relative to meal")
plt.ylabel("Glucose (mg/dL)")
plt.title("Glucose ±2 Hours Around Meals")
plt.grid(True)
plt.legend()
plt.show()