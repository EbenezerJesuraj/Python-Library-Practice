import pandas as pd

df = pd.read_csv('sample_data/california_housing_train.csv') # Loading data into pandas DataFrame from csv

print(df.head()) # First 5 rows
print(df.info()) # Group by Column Non-Null Values and their Dtype
print(df.describe())  # Statistical summary of the data
