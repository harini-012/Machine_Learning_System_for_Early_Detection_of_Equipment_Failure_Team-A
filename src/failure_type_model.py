"""
Advanced Task T38 – Specific Failure-Type Multiclass Prediction Model
Expected Submission: Failure-type model (models/failure_type_model.joblib)

Trains and evaluates a multiclass classifier that predicts specific AI4I 2020
equipment failure modes instead of only binary overall failure:
  - No Failure
  - HDF (Heat Dissipation Failure)
  - OSF (Overstrain Failure)
  - PWF (Power Failure)
  - TWF (Tool Wear Failure)
  - RNF (Random / Other Failure)
"""

import json
from pathlib import Path
from typing import Dict, Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw" / "ai4i2020.csv"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

FAILURE_CLASSES = [
    "No Failure",
    "HDF (Heat Dissipation)",
    "OSF (Overstrain)",
    "PWF (Power Failure)",
    "TWF (Tool Wear)",
    "RNF (Random Failure)",
]

SHORT_CLASS_NAMES = ["No Failure", "HDF", "OSF", "PWF", "TWF", "RNF"]

FEATURE_COLUMNS = [
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

XGB_FEATURE_COLUMNS = [
    c.replace("[", "").replace("]", "") for c in FEATURE_COLUMNS
]


def sanitize_for_xgb(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with brackets removed from column names for XGBoost compatibility."""
    out = df.copy()
    out.columns = [str(c).replace("[", "").replace("]", "") for c in out.columns]
    return out[XGB_FEATURE_COLUMNS]


def derive_failure_type_label(row: pd.Series) -> int:
    """Assign each equipment record to its primary specific failure class (0..5)."""
    if int(row.get("HDF", 0)) == 1:
        return 1
    if int(row.get("OSF", 0)) == 1:
        return 2
    if int(row.get("PWF", 0)) == 1:
        return 3
    if int(row.get("TWF", 0)) == 1:
        return 4
    if int(row.get("RNF", 0)) == 1 or int(row.get("Machine failure", 0)) == 1:
        return 5
    return 0


def prepare_multiclass_dataset():
    """Prepare the 9-feature matrix and aligned binary + multiclass targets using the project split."""
    df = pd.read_csv(RAW_DATA_PATH)
    df["failure_type_id"] = df.apply(derive_failure_type_label, axis=1)

    sensor_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    X = df[sensor_cols + ["Type"]].copy()
    X["Temperature Difference [K]"] = (
        X["Process temperature [K]"] - X["Air temperature [K]"]
    )
    y_binary = df["Machine failure"].astype(int)
    y_multi = df["failure_type_id"].astype(int)

    # Use identical random_state=42 and stratify=y_binary to align 1:1 with X_train.csv / X_test.csv
    X_train_raw, X_test_raw, y_bin_train, y_bin_test, y_multi_train, y_multi_test = (
        train_test_split(
            X,
            y_binary,
            y_multi,
            test_size=0.20,
            random_state=42,
            stratify=y_binary,
        )
    )

    X_train = pd.get_dummies(X_train_raw, columns=["Type"], prefix="Type", dtype=int)
    X_test = pd.get_dummies(X_test_raw, columns=["Type"], prefix="Type", dtype=int)
    X_train = X_train.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    X_test = X_test.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    return X_train, X_test, y_bin_train, y_bin_test, y_multi_train, y_multi_test


def train_and_save_failure_type_model() -> Dict[str, Any]:
    """Train multiclass XGBoost & Random Forest failure-type models, save best model and reports."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, _, _, y_train, y_test = prepare_multiclass_dataset()
    X_train_xgb = sanitize_for_xgb(X_train)
    X_test_xgb = sanitize_for_xgb(X_test)

    # Moderated sample weights to boost minority failure classes without excessive false alarms
    raw_weights = compute_sample_weight("balanced", y_train)
    sample_weights = np.power(raw_weights, 0.55)

    xgb_multi = XGBClassifier(
        objective="multi:softprob",
        num_class=len(FAILURE_CLASSES),
        n_estimators=260,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.90,
        min_child_weight=2,
        random_state=42,
        eval_metric="mlogloss",
    )
    xgb_multi.fit(X_train_xgb, y_train, sample_weight=sample_weights)

    rf_multi = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    rf_multi.fit(X_train_xgb, y_train)

    # Evaluate both candidates on held-out test set
    xgb_pred = xgb_multi.predict(X_test_xgb)
    rf_pred = rf_multi.predict(X_test_xgb)

    xgb_macro_f1 = f1_score(y_test, xgb_pred, average="macro", zero_division=0)
    rf_macro_f1 = f1_score(y_test, rf_pred, average="macro", zero_division=0)

    best_model = xgb_multi if xgb_macro_f1 >= rf_macro_f1 else rf_multi
    best_name = "XGBClassifier (Multiclass)" if xgb_macro_f1 >= rf_macro_f1 else "RandomForestClassifier (Multiclass)"
    y_pred = xgb_pred if xgb_macro_f1 >= rf_macro_f1 else rf_pred
    y_proba = best_model.predict_proba(X_test_xgb)

    # Save trained failure-type model and class mapping
    model_path = MODELS_DIR / "failure_type_model.joblib"
    classes_path = MODELS_DIR / "failure_type_classes.json"
    joblib.dump(best_model, model_path)

    class_metadata = {
        "model_name": best_name,
        "feature_columns": XGB_FEATURE_COLUMNS,
        "classes": {str(i): name for i, name in enumerate(FAILURE_CLASSES)},
        "short_classes": {str(i): name for i, name in enumerate(SHORT_CLASS_NAMES)},
        "metrics": {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "weighted_precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "weighted_recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "weighted_f1": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "macro_precision": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "macro_recall": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        },
    }
    classes_path.write_text(json.dumps(class_metadata, indent=2), encoding="utf-8")

    # Save per-class classification report CSV
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(FAILURE_CLASSES))),
        target_names=FAILURE_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    report_rows = []
    for cls_name in FAILURE_CLASSES + ["macro avg", "weighted avg"]:
        if cls_name in report_dict:
            m = report_dict[cls_name]
            report_rows.append(
                {
                    "failure_class": cls_name,
                    "precision": round(float(m["precision"]), 4),
                    "recall": round(float(m["recall"]), 4),
                    "f1_score": round(float(m["f1-score"]), 4),
                    "support": int(m["support"]),
                }
            )
    report_df = pd.DataFrame(report_rows)
    report_csv_path = REPORTS_DIR / "t38_failure_type_model_report.csv"
    report_df.to_csv(report_csv_path, index=False)

    # Plot 6x6 confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(FAILURE_CLASSES))))
    fig, ax = plt.subplots(figsize=(8, 6.2))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=SHORT_CLASS_NAMES,
        yticklabels=SHORT_CLASS_NAMES,
        cbar=False,
        ax=ax,
    )
    ax.set_title(f"T38: Multiclass Failure-Type Confusion Matrix ({best_name})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Failure Type", fontsize=10)
    ax.set_ylabel("Actual Failure Type", fontsize=10)
    plt.tight_layout()
    cm_path = REPORTS_DIR / "t38_failure_type_confusion_matrix.png"
    fig.savefig(cm_path, dpi=160)
    plt.close(fig)

    return {
        "model_path": str(model_path),
        "report_csv_path": str(report_csv_path),
        "confusion_matrix_path": str(cm_path),
        "metadata": class_metadata,
        "per_class_report": report_rows,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def predict_specific_failure_type(readings: Dict[str, Any]) -> Dict[str, Any]:
    """Predict the specific failure type and class probabilities for a single sensor observation."""
    model_path = MODELS_DIR / "failure_type_model.joblib"
    if not model_path.exists():
        train_and_save_failure_type_model()
    model = joblib.load(model_path)

    eq_type = str(readings.get("Type", readings.get("equipment_type", "L"))).strip().upper()
    air = float(readings.get("Air temperature [K]", readings.get("air_temperature", 300.0)))
    proc = float(readings.get("Process temperature [K]", readings.get("process_temperature", 310.0)))
    speed = float(readings.get("Rotational speed [rpm]", readings.get("rotational_speed", 1500.0)))
    torque = float(readings.get("Torque [Nm]", readings.get("torque", 40.0)))
    wear = float(readings.get("Tool wear [min]", readings.get("tool_wear", 0.0)))

    row_df = pd.DataFrame(
        [
            {
                "Air temperature [K]": air,
                "Process temperature [K]": proc,
                "Rotational speed [rpm]": speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": wear,
                "Temperature Difference [K]": proc - air,
                "Type_H": 1 if eq_type == "H" else 0,
                "Type_L": 1 if eq_type == "L" else 0,
                "Type_M": 1 if eq_type == "M" else 0,
            }
        ]
    )[FEATURE_COLUMNS]

    row_xgb = sanitize_for_xgb(row_df)
    probs = model.predict_proba(row_xgb)[0]
    pred_idx = int(np.argmax(probs))
    failure_mode_idx = int(np.argmax(probs[1:]) + 1)

    return {
        "predicted_class_id": pred_idx,
        "predicted_failure_type": FAILURE_CLASSES[pred_idx],
        "predicted_failure_code": SHORT_CLASS_NAMES[pred_idx],
        "top_specific_failure_mode": FAILURE_CLASSES[failure_mode_idx],
        "top_specific_failure_prob": round(float(probs[failure_mode_idx]), 4),
        "class_probabilities": {
            SHORT_CLASS_NAMES[i]: round(float(probs[i]), 4)
            for i in range(len(SHORT_CLASS_NAMES))
        },
    }


if __name__ == "__main__":
    res = train_and_save_failure_type_model()
    print("T38 Failure-Type Model trained and saved:", res["model_path"])
    print("Overall Multiclass Metrics:", json.dumps(res["metadata"]["metrics"], indent=2))
