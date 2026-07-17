import pandas as pd

file = r"D:\FYP\DATA_SET\cgmacros-a-scientific-dataset-for-personalized-nutrition-and-diet-monitoring-1.0.0\CGMacros_dateshifted365\CGMacros\CGMacros-001\CGMacros-001.csv"

df = pd.read_csv(file)

print(df.columns.tolist())
print(df.head())