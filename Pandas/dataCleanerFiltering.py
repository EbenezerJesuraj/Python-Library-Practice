import numpy as np
import pandas as pd

# Handling missing Data
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, np.nan, 30, 35],
        'City': ['NY', 'LA', np.nan, 'CH']}

df = pd.DataFrame(data)

df_cleaned = df.dropna()

print(df_cleaned)

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['City'] = df['City'].fillna("Unknown")

print(df)

df['City'] = df['City'].str.upper()

print(df)

df_filtered = df[df['Age'] > 28]

print(df_filtered)

fortune_df = pd.read_csv("/content/drive/MyDrive/Python - Colab Files/fortune1000.csv")

# Removing commas and dollar signs, then converting to float
fortune_df['Revenue'] = fortune_df['Revenue'].astype(str).str.replace(',', '').str.replace('$', '').astype(float)

print(fortune_df.head())
