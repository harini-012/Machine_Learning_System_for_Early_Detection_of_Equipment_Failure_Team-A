from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
SELECTED_FEATURES_PATH = REPORTS_DIR / "selected_features.csv"


# --------------------------------------------------
# 2. Load trained model and feature schema
# --------------------------------------------------

def load_best_model(model_path=BEST_MODEL_PATH):
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Best model not found at {model_path}. "
            "Run src/evaluate_models.py or src/tune_model.py first."
        )
    return joblib.load(model_path)


def load_selected_features(features_path=SELECTED_FEATURES_PATH):
    if Path(features_path).exists():
        df = pd.read_csv(features_path)
        return df["feature"].tolist()
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
# 3. Preprocess raw or engineered sensor input
# --------------------------------------------------

def prepare_features(input_data, feature_columns=None):
    """
    Accepts a dictionary, list of dictionaries, or pandas DataFrame.
    Applies feature engineering (Temperature Difference and Type encoding)
    if raw sensor columns are provided, and aligns with the model schema.
    """
    if feature_columns is None:
        feature_columns = load_selected_features()

    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise TypeError("input_data must be a dict, list of dicts, or DataFrame.")

    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]

    # Compute engineered feature if not already present
    if (
        "Temperature Difference [K]" not in df.columns
        and "Process temperature [K]" in df.columns
        and "Air temperature [K]" in df.columns
    ):
        df["Temperature Difference [K]"] = (
            df["Process temperature [K]"] - df["Air temperature [K]"]
        )

    # One-hot encode Type if provided as categorical column
    if "Type" in df.columns:
        df = pd.get_dummies(df, columns=["Type"], prefix="Type", dtype=int)

    # Align with exact training feature columns
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


# --------------------------------------------------
# 4. Predict equipment failure and failure probability
# --------------------------------------------------

def assign_risk_level(probability):
    if probability >= 0.70:
        return "High Risk"
    if probability >= 0.30:
        return "Medium Risk"
    return "Low Risk"


def predict_failure(input_data, model=None, threshold=0.50):
    """
    Generates binary failure predictions, failure probabilities, and
    early-warning risk categories for the provided sensor observations.
    """
    if model is None:
        model = load_best_model()

    feature_columns = load_selected_features()
    X_prepared = prepare_features(input_data, feature_columns)

    probabilities = model.predict_proba(X_prepared)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    risk_levels = [assign_risk_level(p) for p in probabilities]

    results = pd.DataFrame({
        "predicted_failure": predictions,
        "failure_probability": np.round(probabilities, 4),
        "risk_level": risk_levels,
    })

    return results


# --------------------------------------------------
# 5. Test failure-probability predictions
# --------------------------------------------------

def main():
    print("=" * 65)
    print("STEP 1: LOADING SELECTED BEST MODEL FOR INFERENCE")
    print("=" * 65)

    model = load_best_model()
    features = load_selected_features()
    print(f"Loaded model type : {type(model).__name__}")
    print(f"Expected features : {features}\n")

    print("=" * 65)
    print("STEP 2: TESTING FAILURE-PROBABILITY PREDICTIONS ON TEST SET")
    print("=" * 65)

    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0].astype(int)

    test_results = predict_failure(X_test, model=model)
    test_results["actual_failure"] = y_test.to_numpy()

    total_samples = len(test_results)
    predicted_failures = int(test_results["predicted_failure"].sum())
    actual_failures = int(test_results["actual_failure"].sum())

    avg_prob_normal = test_results.loc[
        test_results["actual_failure"] == 0, "failure_probability"
    ].mean()
    avg_prob_failure = test_results.loc[
        test_results["actual_failure"] == 1, "failure_probability"
    ].mean()

    print(f"Total test samples evaluated         : {total_samples}")
    print(f"Actual failures in test set          : {actual_failures}")
    print(f"Predicted failures (threshold=0.50)  : {predicted_failures}")
    print(f"Mean probability for Normal class (0): {avg_prob_normal:.4f}")
    print(f"Mean probability for Failure class(1): {avg_prob_failure:.4f}\n")

    print("Risk Level Distribution on Test Set:")
    print(test_results["risk_level"].value_counts().to_string())

    print("\n" + "=" * 65)
    print("STEP 3: SAMPLE EARLY-WARNING SCENARIO PREDICTIONS")
    print("=" * 65)

    sample_scenarios = [
        {
            "Scenario": "Normal Operating Condition",
            "Type": "M",
            "Air temperature [K]": 298.1,
            "Process temperature [K]": 308.6,
            "Rotational speed [rpm]": 1550,
            "Torque [Nm]": 40.0,
            "Tool wear [min]": 45,
        },
        {
            "Scenario": "Elevated Tool Wear & High Torque (Overstrain Risk)",
            "Type": "L",
            "Air temperature [K]": 300.5,
            "Process temperature [K]": 310.2,
            "Rotational speed [rpm]": 1320,
            "Torque [Nm]": 62.5,
            "Tool wear [min]": 215,
        },
        {
            "Scenario": "Low Speed & Small Temp Difference (Heat Dissipation Risk)",
            "Type": "L",
            "Air temperature [K]": 302.8,
            "Process temperature [K]": 310.4,
            "Rotational speed [rpm]": 1310,
            "Torque [Nm]": 54.0,
            "Tool wear [min]": 120,
        },
        {
            "Scenario": "Extreme Power Overload Condition",
            "Type": "H",
            "Air temperature [K]": 299.2,
            "Process temperature [K]": 309.1,
            "Rotational speed [rpm]": 1280,
            "Torque [Nm]": 70.0,
            "Tool wear [min]": 185,
        },
    ]

    scenario_df = pd.DataFrame(sample_scenarios)
    scenario_preds = predict_failure(scenario_df.drop(columns=["Scenario"]), model=model)
    scenario_summary = pd.concat([scenario_df[["Scenario"]], scenario_preds], axis=1)
    print(scenario_summary.to_string(index=False))


if __name__ == "__main__":
    main()
