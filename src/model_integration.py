
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "best_model.joblib"
FEATURES_PATH = PROJECT_DIR / "reports" / "selected_features.csv"
TEST_DATA_PATH = PROJECT_DIR / "data" / "processed" / "X_test.csv"
TEST_LABEL_PATH = PROJECT_DIR / "data" / "processed" / "y_test.csv"

REPORTS_DIR = PROJECT_DIR / "reports"
PREDICTION_OUTPUT = REPORTS_DIR / "prediction_results.csv"


# --------------------------------------------------
# 2. Load the trained model
# --------------------------------------------------

def load_best_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The selected model does not support predict_proba()."
        )

    print("Best model loaded:", type(model).__name__)

    return model


# --------------------------------------------------
# 3. Load the selected feature names
# --------------------------------------------------

def load_feature_columns():

    if FEATURES_PATH.exists():
        feature_data = pd.read_csv(FEATURES_PATH)
        return feature_data["feature"].tolist()

    return [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Temperature Difference [K]",
        "Type_H",
        "Type_L",
        "Type_M",
    ]


# --------------------------------------------------
# 4. Prepare sensor readings
# --------------------------------------------------

def prepare_features(input_data, feature_columns):

    # Accept a dictionary, list of dictionaries or DataFrame.
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])

    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)

    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()

    else:
        raise TypeError(
            "Input must be a dictionary, list of dictionaries or DataFrame."
        )

    if df.empty:
        raise ValueError("No sensor readings were provided.")

    # Keep equipment ID separately; it is not a model input.
    if "equipment_id" in df.columns:
        equipment_ids = df["equipment_id"].astype(str).tolist()
        df = df.drop(columns=["equipment_id"])

    elif "Equipment ID" in df.columns:
        equipment_ids = df["Equipment ID"].astype(str).tolist()
        df = df.drop(columns=["Equipment ID"])

    else:
        # Generated observation IDs for demonstration/test data.
        equipment_ids = [
            f"OBS-{i + 1:06d}" for i in range(len(df))
        ]

    # Remove identifiers and target columns if supplied.
    columns_to_remove = [
        "UDI",
        "Product ID",
        "Machine failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]

    df = df.drop(
        columns=columns_to_remove,
        errors="ignore"
    )

    # Remove accidental CSV index columns.
    df = df.loc[
        :, ~df.columns.astype(str).str.startswith("Unnamed:")
    ]

    # Calculate temperature difference when raw temperatures exist.
    if (
        "Temperature Difference [K]" not in df.columns
        and "Process temperature [K]" in df.columns
        and "Air temperature [K]" in df.columns
    ):
        df["Temperature Difference [K]"] = (
            df["Process temperature [K]"]
            - df["Air temperature [K]"]
        )

    # Convert equipment type into numerical columns.
    if "Type" in df.columns:
        if not df["Type"].isin(["L", "M", "H"]).all():
            raise ValueError("Type must contain L, M or H.")

        df = pd.get_dummies(
            df,
            columns=["Type"],
            prefix="Type",
            dtype=int
        )

    # Match the exact training feature order.
    # Match the exact original training feature order.
        df = df.reindex(
    columns=feature_columns,
    fill_value=0
)

# Convert feature names to the format expected by XGBoost.
        df.columns = df.columns.str.replace(
    r"[\[\]<]", "", regex=True
)

    # Convert original names to the names expected by XGBoost.
    df.columns = (
        df.columns
        .str.replace(r"[\[\]<]", "", regex=True)
    )

    # Ensure every feature is numeric.
    df = df.apply(pd.to_numeric, errors="coerce")

    if df.isnull().any().any():
        raise ValueError(
            "Sensor input contains missing or non-numeric feature values."
        )

    return df, equipment_ids


# --------------------------------------------------
# 5. Predict equipment failure
# --------------------------------------------------

def predict_failure(input_data, model=None, threshold=0.50):

    if model is None:
        model = load_best_model()

    feature_columns = load_feature_columns()

    X, equipment_ids = prepare_features(
        input_data,
        feature_columns
    )

    # Check that the prepared names match the saved model.
    expected_names = getattr(model, "feature_names_in_", None)

    if expected_names is not None:
        if list(X.columns) != list(expected_names):
            raise ValueError(
                "Prepared feature names do not match the model's expected features."
            )

    # Calculate probability of failure (class 1).
    probabilities = model.predict_proba(X)[:, 1]

    # Convert probability into binary prediction.
    predictions = (probabilities >= threshold).astype(int)

    results = pd.DataFrame({
        "equipment_id": equipment_ids,
        "predicted_failure": predictions,
        "failure_probability": np.round(probabilities, 4),
        "prediction_label": [
            "Failure" if value == 1 else "No Failure"
            for value in predictions
        ],
        "prediction_timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
    })

    return results


# --------------------------------------------------
# 6. Save prediction results to CSV
# --------------------------------------------------

def save_prediction_results(results):

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        PREDICTION_OUTPUT,
        mode="a",
        header=not PREDICTION_OUTPUT.exists(),
        index=False
    )

    print("Prediction results saved to:", PREDICTION_OUTPUT)


# --------------------------------------------------
# 7. Test the prediction module
# --------------------------------------------------

def main():

    print("\nSTEP 1: Loading the best model")

    model = load_best_model()
    sample_input = {
    "equipment_id": "EQ-001",
    "Type": "L",
    "Air temperature [K]": 300.5,
    "Process temperature [K]": 310.2,
    "Rotational speed [rpm]": 1320,
    "Torque [Nm]": 62.5,
    "Tool wear [min]": 215
}

    sample_result = predict_failure(
    sample_input,
    model=model
)

    print("\nSample equipment prediction:")
    print(sample_result.to_string(index=False))
    print("\nSTEP 2: Loading test sensor readings")

    X_test = pd.read_csv(TEST_DATA_PATH)

    y_test = pd.read_csv(TEST_LABEL_PATH).iloc[:, 0].astype(int)

    print("Test records:", len(X_test))

    print("\nSTEP 3: Predicting equipment failures")

    results = predict_failure(
        X_test,
        model=model
    )

    results["actual_failure"] = y_test.to_numpy()

    print("\nPrediction results:")
    print(results.head(10).to_string(index=False))

    print("\nSTEP 4: Prediction summary")

    print("Total records:", len(results))

    print(
        "Predicted failures:",
        int(results["predicted_failure"].sum())
    )

    print(
        "Actual failures:",
        int(results["actual_failure"].sum())
    )

    print(
        "Average failure probability:",
        round(results["failure_probability"].mean(), 4)
    )

    print("\nSTEP 5: Saving results")

    save_prediction_results(results)

    print("\nMember 1 prediction module completed successfully.")


if __name__ == "__main__":
    main()
