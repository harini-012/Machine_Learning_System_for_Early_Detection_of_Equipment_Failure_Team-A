import pandas as pd
from pathlib import Path

# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = PROJECT_DIR / "data" / "raw" / "ai4i2020.csv"
SENSOR_FILE = PROJECT_DIR / "data" / "processed" / "sensor_readings.csv"
PREDICTION_FILE = PROJECT_DIR / "reports" / "prediction_results.csv"
OUTPUT_FILE = PROJECT_DIR / "reports" / "anomaly_results.csv"

# ==========================================
# 2. SENSOR COLUMNS
# ==========================================

SENSORS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# ==========================================
# 3. LOAD SENSOR DATA
# ==========================================

print("\nLoading sensor readings...")

if SENSOR_FILE.exists():
    df = pd.read_csv(SENSOR_FILE)
    print("Processed sensor readings loaded.")

elif RAW_FILE.exists():
    df = pd.read_csv(RAW_FILE)
    print("Original AI4I dataset loaded.")

else:
    raise FileNotFoundError("Sensor dataset not found.")

# ==========================================
# 4. STANDARDIZE EQUIPMENT ID
# ==========================================

if "equipment_id" in df.columns:
    df.rename(
        columns={"equipment_id": "Equipment ID"},
        inplace=True
    )

elif "UDI" in df.columns:
    df["Equipment ID"] = (
        "OBS-" +
        df["UDI"].astype(int).astype(str).str.zfill(6)
    )

if "Equipment ID" not in df.columns:
    raise ValueError("Equipment ID column not found.")

df["Equipment ID"] = df["Equipment ID"].astype(str).str.strip()

# ==========================================
# 5. VALIDATE SENSOR DATA
# ==========================================

print("\nChecking sensor columns...")

missing = [s for s in SENSORS if s not in df.columns]

if missing:
    raise ValueError(f"Missing sensor columns: {missing}")

for sensor in SENSORS:
    df[sensor] = pd.to_numeric(df[sensor], errors="coerce")

if df[SENSORS].isnull().any().any():
    raise ValueError("Missing or invalid sensor readings.")

print("All five sensor columns verified.")

# ==========================================
# 6. LOAD PREDICTION RESULTS
# ==========================================

print("\nLoading Member 1 predictions...")

if not PREDICTION_FILE.exists():
    raise FileNotFoundError("Prediction CSV not found.")

predictions = pd.read_csv(PREDICTION_FILE)

predictions.rename(
    columns={
        "equipment_id": "Equipment ID",
        "failure_probability": "Failure Probability"
    },
    inplace=True
)

required = ["Equipment ID", "Failure Probability"]

for col in required:
    if col not in predictions.columns:
        raise ValueError(f"Missing prediction column: {col}")

predictions["Equipment ID"] = (
    predictions["Equipment ID"].astype(str).str.strip()
)

predictions["Failure Probability"] = pd.to_numeric(
    predictions["Failure Probability"],
    errors="coerce"
)

if predictions["Failure Probability"].isna().any():
    raise ValueError("Invalid failure probability.")

if not predictions["Failure Probability"].between(0, 1).all():
    raise ValueError("Probability must be between 0 and 1.")

# ==========================================
# 7. REMOVE DUPLICATE PREDICTIONS
# ==========================================

print("\nChecking duplicate equipment IDs...")

duplicate_count = predictions["Equipment ID"].duplicated().sum()

print("Duplicate prediction records:", duplicate_count)

if duplicate_count > 0:

    if "prediction_timestamp" in predictions.columns:

        predictions["prediction_timestamp"] = pd.to_datetime(
            predictions["prediction_timestamp"],
            errors="coerce",
            utc=True
        )

        predictions = predictions.sort_values(
            "prediction_timestamp",
            na_position="first"
        )

    predictions = predictions.drop_duplicates(
        subset=["Equipment ID"],
        keep="last"
    )

print("Unique prediction records:", len(predictions))

# ==========================================
# 8. MATCH AVAILABLE EQUIPMENT RECORDS
# ==========================================

print("\nMatching sensor readings with predictions...")

sensor_count = len(df)

prediction_ids = set(predictions["Equipment ID"])

matching_ids = set(df["Equipment ID"]) & prediction_ids

print("Total sensor records:", sensor_count)
print("Matching equipment IDs:", len(matching_ids))

# Keep only sensor records having predictions

df = df.merge(
    predictions[["Equipment ID", "Failure Probability"]],
    on="Equipment ID",
    how="inner",
    validate="many_to_one"
)

excluded = sensor_count - len(df)

print("Unmatched sensor records excluded:", excluded)
print("Matched records ready for analysis:", len(df))

if df.empty:
    raise ValueError("No matching equipment IDs found.")

# ==========================================
# 9. PREPARE NORMAL REFERENCE DATA
# ==========================================

print("\nPreparing normal reference data...")

if RAW_FILE.exists():

    reference = pd.read_csv(RAW_FILE)

    normal_data = reference[
        reference["Machine failure"] == 0
    ].copy()

else:
    normal_data = df.copy()

for sensor in SENSORS:
    normal_data[sensor] = pd.to_numeric(
        normal_data[sensor],
        errors="coerce"
    )

normal_data = normal_data.dropna(subset=SENSORS)

if normal_data.empty:
    raise ValueError("Normal reference data is empty.")

# ==========================================
# 10. CALCULATE IQR THRESHOLDS
# ==========================================

print("\nCalculating IQR thresholds...")

limits = {}

for sensor in SENSORS:

    q1 = normal_data[sensor].quantile(0.25)
    q3 = normal_data[sensor].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    limits[sensor] = (lower, upper)

    print(f"{sensor}: {lower:.3f} to {upper:.3f}")

# ==========================================
# 11. DETECT ABNORMAL SENSOR READINGS
# ==========================================

print("\nDetecting abnormal sensor readings...")

anomaly_columns = []

for sensor in SENSORS:

    lower, upper = limits[sensor]

    anomaly_column = sensor + " Anomaly"

    df[anomaly_column] = (
        (df[sensor] < lower) |
        (df[sensor] > upper)
    )

    anomaly_columns.append(anomaly_column)

# ==========================================
# 12. COUNT ANOMALIES
# ==========================================

df["Anomaly Count"] = (
    df[anomaly_columns].sum(axis=1)
)

# ==========================================
# 13. IDENTIFY ABNORMAL SENSORS
# ==========================================

def identify_anomalies(row):

    abnormal = []

    for sensor in SENSORS:

        if row[sensor + " Anomaly"]:
            abnormal.append(sensor)

    return ", ".join(abnormal) if abnormal else "None"


df["Abnormal Sensors"] = df.apply(
    identify_anomalies,
    axis=1
)

# ==========================================
# 14. CLASSIFY EQUIPMENT RISK
# ==========================================

def classify_risk(row):

    probability = row["Failure Probability"]
    anomalies = row["Anomaly Count"]

    if probability >= 0.70 or anomalies >= 2:
        return "Critical"

    elif probability >= 0.30 or anomalies == 1:
        return "Warning"

    else:
        return "Healthy"


df["Risk Level"] = df.apply(
    classify_risk,
    axis=1
)

# ==========================================
# 15. GENERATE RISK SUMMARY
# ==========================================

def generate_summary(row):

    risk = row["Risk Level"]
    abnormal = row["Abnormal Sensors"]

    if risk == "Critical":

        return (
            "Critical equipment risk. "
            "Detailed equipment inspection recommended. "
            "Abnormal sensors: " + abnormal
        )

    elif risk == "Warning":

        return (
            "Equipment requires monitoring. "
            "Check abnormal sensor readings. "
            "Abnormal sensors: " + abnormal
        )

    else:

        return (
            "Healthy equipment condition. "
            "No significant abnormal sensor readings detected."
        )


df["Risk Summary"] = df.apply(
    generate_summary,
    axis=1
)

# ==========================================
# 16. SAVE ANOMALY RESULTS
# ==========================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(OUTPUT_FILE, index=False)

# ==========================================
# 17. DISPLAY FINAL RESULTS
# ==========================================

print("\n======================================")
print("ANOMALY DETECTION COMPLETED")
print("======================================")

print("\nTotal matched equipment:", len(df))

print("\nRisk Distribution:")

print(df["Risk Level"].value_counts())

print("\nAnomaly Distribution:")

print(df["Anomaly Count"].value_counts().sort_index())

print("\nEquipment Analysis:")

print(
    df[
        [
            "Equipment ID",
            "Failure Probability",
            "Anomaly Count",
            "Abnormal Sensors",
            "Risk Level",
            "Risk Summary"
        ]
    ].head(10).to_string(index=False)
)

print("\n======================================")
print("OUTPUT SAVED SUCCESSFULLY")
print(OUTPUT_FILE)
print("======================================")