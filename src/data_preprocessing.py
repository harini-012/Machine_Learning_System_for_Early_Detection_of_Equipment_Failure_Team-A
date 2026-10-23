
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_DIR / "data" / "raw" / "ai4i2020.csv"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "equipment_cleaned.csv"
REPORTS_DIR = PROJECT_DIR / "reports"

# Create output folders if needed
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

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

# ==========================================
# Outlier detection using the IQR method
# ==========================================

sensor_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

outlier_results = []

print("\n--- Outlier Analysis ---")

for column in sensor_columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    count = int(mask.sum())

    outlier_results.append({
        "Sensor": column,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower_Bound": lower_bound,
        "Upper_Bound": upper_bound,
        "Outlier_Count": count,
        "Outlier_Percentage": round(mask.mean() * 100, 2)
    })

    print(f"{column}: {count} potential outliers")

# Save the outlier report
outlier_report_path = REPORTS_DIR / "outlier_analysis.csv"
pd.DataFrame(outlier_results).to_csv(
    outlier_report_path, index=False
)

print("Outlier report saved to:", outlier_report_path)

# Save cleaned dataset
# Original sensor readings are preserved; outliers are not
# automatically removed or modified.
df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset shape:", df.shape)
print("Saved cleaned dataset to:", OUTPUT_PATH)
