import pandas as pd

dfc = pd.read_csv("data/processed/CHANDRA_filtered.csv")
print(dfc.head())

print("Filtered data shape:", dfc.shape)

print(dfc.info())