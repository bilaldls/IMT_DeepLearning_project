import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/CHANDRA.csv")
OUTPUT_PATH = Path("data/processed/CHANDRA_filtered.csv")

df = pd.read_csv(INPUT_PATH)
print("Initial data shape:", df.shape)
print(df.info())

# Keep columns 6 to 18 (1-indexed) -> pandas iloc uses 0-index
df = df.iloc[:, 6:19]
print("After column selection:", df.shape)

# Keep only non-negative delta_t_min rows
df = df[df["delta_t_min"].ge(0)]
print("After filtering delta_t_min >= 0:", df.shape)

# Drop rows with any missing values
df = df.dropna()
print("After dropping missing values:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()
print("After dropping duplicates:", df.shape)

df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved cleaned dataset to {OUTPUT_PATH}")
