
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "equipment_cleaned.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

# Load cleaned data
df = pd.read_csv(DATA_PATH)

# Remove identifiers, target, and failure-mode indicators
columns_to_drop = [
    "UDI",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
]

X = df.drop(columns=columns_to_drop, errors="raise")
y = df["Machine failure"].copy()

# Convert categorical equipment Type into numerical columns
X = pd.get_dummies(
    X,
    columns=["Type"],
    dtype=int
)

# Check that features and target are valid
if X.isnull().any().any():
    raise ValueError("Missing values remain in the feature dataset.")

if y.isnull().any():
    raise ValueError("Missing values remain in the target.")

if not set(y.unique()).issubset({0, 1}):
    raise ValueError("Machine failure must contain only 0 and 1.")

# Save features and target
X.to_csv(OUTPUT_DIR / "equipment_features.csv", index=False)
y.to_frame().to_csv(OUTPUT_DIR / "equipment_target.csv", index=False)

# Save feature names
pd.Series(X.columns, name="feature").to_csv(
    OUTPUT_DIR / "feature_columns.csv",
    index=False
)

print("Feature preparation completed.")
print("Number of records:", len(X))
print("Number of input features:", X.shape[1])
print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

print("\nSaved features and target in:", OUTPUT_DIR)
