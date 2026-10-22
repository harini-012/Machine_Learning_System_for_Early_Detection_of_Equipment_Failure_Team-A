
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_DIR / "data" / "raw" / "ai4i2020.csv"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "equipment_cleaned.csv"

# Create output folder if needed
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load original dataset
df = pd.read_csv(RAW_PATH)

print("Original dataset shape:", df.shape)

# Remove duplicate rows
duplicates = df.duplicated().sum()
print("Duplicate rows found:", duplicates)

df = df.drop_duplicates().copy()

# Identify column types
numeric_columns = df.select_dtypes(include="number").columns
categorical_columns = df.select_dtypes(
    include=["object", "category", "bool"]
).columns

# Handle missing numerical values using the median
for column in numeric_columns:
    if df[column].isnull().any():
        df[column] = df[column].fillna(df[column].median())

# Handle missing categorical values using the mode
for column in categorical_columns:
    if df[column].isnull().any():
        mode_values = df[column].mode()
        if not mode_values.empty:
            df[column] = df[column].fillna(mode_values.iloc[0])

# Check remaining missing values
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Save cleaned dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset shape:", df.shape)
print("Saved cleaned dataset to:", OUTPUT_PATH)
