
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent#Going to check the python file

DATA_PATH = PROJECT_DIR / "data" / "processed" / "equipment_cleaned.csv"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORTS_DIR = PROJECT_DIR / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load and verify the cleaned dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())

required_columns = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

if df[required_columns].isnull().any().any():
    raise ValueError(
        "Required columns contain missing values. "
        "Resolve them before feature engineering."
    )

if not set(df["Machine failure"].unique()).issubset({0, 1}):
    raise ValueError("Machine failure must contain only 0 and 1.")

if df["Machine failure"].nunique() != 2:
    raise ValueError("Both failure classes must be present.")

# --------------------------------------------------
# 3. Select features and target
# --------------------------------------------------

sensor_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

X = df[sensor_columns + ["Type"]].copy()#input feature considering the sensor columns
y = df["Machine failure"].astype(int).copy()#output feature considering the machine failiure

# --------------------------------------------------
# 4. Feature engineering
# --------------------------------------------------
#Why create this feature? The difference between process and air temperatures may provide useful information about operating conditions.
X["Temperature Difference [K]"] = (
    X["Process temperature [K]"]
    - X["Air temperature [K]"]
)

print("\nFeatures before encoding:")
print(X.columns.tolist())

# --------------------------------------------------
# 5. Split before fitting any learned preprocessing
# --------------------------------------------------

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# --------------------------------------------------
# 6. Encode Type using one-hot encoding
#Your code converts these categories into numeric indicator columns.
# --------------------------------------------------

X_train = pd.get_dummies(
    X_train_raw,
    columns=["Type"],
    prefix="Type",
    dtype=int,
)

X_test = pd.get_dummies(
    X_test_raw,
    columns=["Type"],
    prefix="Type",
    dtype=int,
)

# Ensure train and test contain exactly the same columns.
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0,
)

# --------------------------------------------------
# 7. Save training and testing datasets
# --------------------------------------------------

X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)

y_train.to_frame().to_csv(
    PROCESSED_DIR / "y_train.csv", index=False
)
y_test.to_frame().to_csv(
    PROCESSED_DIR / "y_test.csv", index=False
)

# Save the feature names for the other team members.
pd.DataFrame({
    "feature": X_train.columns
}).to_csv(
    REPORTS_DIR / "selected_features.csv",
    index=False,
)
#X_train.csv: inputs used to train models.



#X_test.csv: inputs used to evaluate models.



#y_train.csv: correct answers for training.



#y_test.csv: correct answers for testing.
# --------------------------------------------------
# 8. Print verification results
# --------------------------------------------------

print("\nFeature engineering completed successfully.")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

print("\nTraining failure distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting failure distribution:")
print(y_test.value_counts().sort_index())

print("\nFinal feature columns:")
print(X_train.columns.tolist())

print("\nSaved files in:", PROCESSED_DIR)
print("Saved feature list in:", REPORTS_DIR)
