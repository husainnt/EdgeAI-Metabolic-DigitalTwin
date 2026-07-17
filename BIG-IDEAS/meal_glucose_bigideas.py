import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------
dexcom_file = r"D:\FYP\DATA_SET\big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.3\001\Dexcom_001.csv"

food_file = r"D:\FYP\DATA_SET\big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.3\001\Food_Log_001.csv"

# -----------------------------
# Load Dexcom
# -----------------------------
dex = pd.read_csv(dexcom_file)

# Keep only glucose records
dex = dex[dex["Event Type"] == "EGV"]

dex["Timestamp"] = pd.to_datetime(
    dex["Timestamp (YYYY-MM-DDThh:mm:ss)"]
)

dex["Glucose"] = dex["Glucose Value (mg/dL)"]

# -----------------------------
# Load Food Log
# -----------------------------
food = pd.read_csv(food_file)

food["MealTime"] = pd.to_datetime(food["time_begin"])

# -----------------------------
# Plot glucose around each meal
# -----------------------------
plt.figure(figsize=(12,6))

for _, meal in food.iterrows():

    t = meal["MealTime"]

    window = dex[
        (dex["Timestamp"] >= t - pd.Timedelta(hours=2)) &
        (dex["Timestamp"] <= t + pd.Timedelta(hours=2))
    ].copy()

    if len(window) < 5:
        continue

    window["Minutes"] = (
        window["Timestamp"] - t
    ).dt.total_seconds()/60

    plt.plot(
        window["Minutes"],
        window["Glucose"],
        alpha=0.4
    )

plt.axvline(
    0,
    color='red',
    linestyle='--',
    label='Meal'
)

plt.xlabel("Minutes relative to meal")
plt.ylabel("Glucose (mg/dL)")
plt.title("BIG IDEAS: Glucose ±2 Hours Around Meals")
plt.grid(True)
plt.legend()

plt.show()