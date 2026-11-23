import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Credit Score" / "data"

# Load dataset
df = pd.read_csv(DATA_DIR / "credit_data.csv")

# Remove ID column
df = df.drop(columns=["customer_id"])

# Convert columns to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# Check missing values
print("Missing values:")
print(df.isnull().sum().sum())

# Check duplicates
print("Duplicate records:", df.duplicated().sum())

# Dataset information
print("\nDataset shape:", df.shape)
print("\nData types:")
print(df.dtypes)

# Target distribution
print("\nTarget distribution:")
print(df["default"].value_counts())

# Save cleaned dataset
df.to_csv(DATA_DIR / "clean_credit_data.csv", index=False)

print("\nPreprocessing completed!")