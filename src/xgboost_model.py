
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load Member 1's prepared datasets
# --------------------------------------------------

def load_datasets():
    paths = {
        "X_train": DATA_DIR / "X_train.csv",
        "X_test": DATA_DIR / "X_test.csv",
        "y_train": DATA_DIR / "y_train.csv",
        "y_test": DATA_DIR / "y_test.csv",
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} dataset not found: {path}"
            )

    X_train = pd.read_csv(paths["X_train"])
    X_test = pd.read_csv(paths["X_test"])

    y_train = pd.read_csv(paths["y_train"]).iloc[:, 0]
    y_test = pd.read_csv(paths["y_test"]).iloc[:, 0]

    # Remove accidental CSV index columns.
    X_train = X_train.loc[
        :, ~X_train.columns.str.startswith("Unnamed:")
    ]
    X_test = X_test.loc[
        :, ~X_test.columns.str.startswith("Unnamed:")
    ]

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Training and testing feature columns do not match."
        )

    if X_train.isnull().any().any() or X_test.isnull().any().any():
        raise ValueError("Missing values found in the feature datasets.")

    if not np.isin(y_train.unique(), [0, 1]).all():
        raise ValueError("Training labels must be encoded as 0 and 1.")

    if not np.isin(y_test.unique(), [0, 1]).all():
        raise ValueError("Testing labels must be encoded as 0 and 1.")

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and target row counts do not match.")

    if set(y_train.unique()) != {0, 1}:
        raise ValueError("Training data must contain both classes.")

    if set(y_test.unique()) != {0, 1}:
        raise ValueError("Testing data must contain both classes.")

    print("\nSTEP 1: DATASET INFORMATION")
    print("-" * 55)
    print("Features:", list(X_train.columns))
    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)
    print("\nTraining label counts:")
    print(y_train.value_counts().sort_index())
    print("\nTesting label counts:")
    print(y_test.value_counts().sort_index())

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# 3. Create validation data for tuning
# --------------------------------------------------

def run_experiments(X_train, y_train):
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train,
    )

    counts = y_fit.value_counts()
    scale_pos_weight = counts.loc[0] / counts.loc[1]

    print("\nSTEP 2: XGBOOST EXPERIMENTS")
    print("-" * 55)
    print("Failure weight:", round(scale_pos_weight, 3))

    # Eight parameter combinations, each with and without
    # positive-class weighting.
    parameter_grid = [
        (100, 3, 0.03),
        (100, 3, 0.10),
        (100, 5, 0.03),
        (100, 5, 0.10),
        (200, 3, 0.03),
        (200, 3, 0.10),
        (200, 5, 0.03),
        (200, 5, 0.10),
    ]

    results = []

    for n_estimators, max_depth, learning_rate in parameter_grid:
        for weighted in [False, True]:

            weight = scale_pos_weight if weighted else 1.0

            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                scale_pos_weight=weight,
                random_state=42,
                n_jobs=-1,
            )

            model.fit(X_fit, y_fit)

            predictions = model.predict(X_val)
            probabilities = model.predict_proba(X_val)[:, 1]

            result = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "class_weighting": weighted,
                "scale_pos_weight": weight,
                "validation_accuracy": accuracy_score(
                    y_val, predictions
                ),
                "validation_precision": precision_score(
                    y_val, predictions, zero_division=0
                ),
                "validation_recall": recall_score(
                    y_val, predictions, zero_division=0
                ),
                "validation_f1": f1_score(
                    y_val, predictions, zero_division=0
                ),
                "validation_roc_auc": roc_auc_score(
                    y_val, probabilities
                ),
            }

            results.append(result)

            print(
                f"Trees={n_estimators}, Depth={max_depth}, "
                f"LR={learning_rate}, Weighted={weighted} | "
                f"F1={result['validation_f1']:.4f}, "
                f"Recall={result['validation_recall']:.4f}"
            )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        ["validation_f1", "validation_recall"],
        ascending=False,
    )

    results_df.to_csv(
        REPORTS_DIR / "xgboost_tuning_results.csv",
        index=False,
    )

    best = results_df.iloc[0]

    best_params = {
        "n_estimators": int(best["n_estimators"]),
        "max_depth": int(best["max_depth"]),
        "learning_rate": float(best["learning_rate"]),
        "scale_pos_weight": float(best["scale_pos_weight"]),
    }

    print("\nSelected validation configuration:")
    print(best_params)

    return best_params


# --------------------------------------------------
# 4. Train final model and evaluate on test data
# --------------------------------------------------

def evaluate_final_model(X_train, X_test, y_train, y_test, params):
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        **params,
    )

    # Fit on the complete training set only after tuning.
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "XGBoost",
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "best_parameters": params,
    }

    print("\nSTEP 3: FINAL TEST EVALUATION")
    print("-" * 55)
    print(json.dumps(metrics, indent=4))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Normal", "Failure"],
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    print("Confusion matrix:")
    print(cm)

    # Save the confusion-matrix plot.
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Failure"],
    )
    display.plot(values_format="d")
    plt.title("XGBoost Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        REPORTS_DIR / "xgboost_confusion_matrix.png",
        dpi=150,
    )
    plt.close()

    # Save feature importance without overwriting Random Forest's file.
    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    importance_df.to_csv(
        REPORTS_DIR / "xgboost_feature_importance.csv",
        index=False,
    )

    # Save model and evaluation metrics.
    joblib.dump(model, MODELS_DIR / "xgboost.joblib")

    with open(
        REPORTS_DIR / "xgboost_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    # Save predictions and failure probabilities.
    pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": predictions,
        "failure_probability": probabilities,
    }).to_csv(
        REPORTS_DIR / "xgboost_predictions.csv",
        index=False,
    )

    print("\nSTEP 4: FILES SAVED")
    print("Model:", MODELS_DIR / "xgboost.joblib")
    print("Tuning:", REPORTS_DIR / "xgboost_tuning_results.csv")
    print("Metrics:", REPORTS_DIR / "xgboost_metrics.json")
    print("Importance:", REPORTS_DIR / "xgboost_feature_importance.csv")
    print("Confusion matrix:", REPORTS_DIR / "xgboost_confusion_matrix.png")
    print("Predictions:", REPORTS_DIR / "xgboost_predictions.csv")


def main():
    X_train, X_test, y_train, y_test = load_datasets()
    params = run_experiments(X_train, y_train)
    evaluate_final_model(
        X_train, X_test, y_train, y_test, params
    )


if __name__ == "__main__":
    main()
