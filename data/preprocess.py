import pandas as pd
import numpy as np


df = pd.read_csv('data/processed/CHANDRA.csv')

print ("Initial data shape:", df.shape)
print( df.info() )

# Drop rows with any missing values
df = df.dropna()
print ("After dropping missing values:", df.shape)
# Remove duplicate rows
df = df.drop_duplicates()
print ("After dropping duplicates:", df.shape)



