import pandas as pd
import matplotlib.pyplot as plt

dex_file = r"D:\FYP\DATA_SET\big-ideas-lab-glycemic-variability-and-wearable-device-data-1.1.3\001\Dexcom_001.csv"

dex = pd.read_csv(dex_file)

dex = dex[dex["Event Type"]=="EGV"]

dex["Timestamp"] = pd.to_datetime(
    dex["Timestamp (YYYY-MM-DDThh:mm:ss)"]
)

dex["Hour"] = dex["Timestamp"].dt.hour

dex["Glucose"] = dex["Glucose Value (mg/dL)"]

hourly = dex.groupby("Hour")["Glucose"].mean()

plt.figure(figsize=(10,5))

plt.plot(
    hourly.index,
    hourly.values,
    marker="o"
)

plt.xticks(range(24))

plt.xlabel("Hour of Day")

plt.ylabel("Average Glucose (mg/dL)")

plt.title("BIG IDEAS: Average Glucose by Hour")

plt.grid(True)

plt.show()