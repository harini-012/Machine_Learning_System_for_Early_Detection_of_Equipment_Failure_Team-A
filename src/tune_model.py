from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load prepared datasets
# --------------------------------------------------

def load_datasets():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0].astype(int)
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0].astype(int)

    X_train = X_train.loc[:, ~X_train.columns.str.startswith("Unnamed:")]
    X_test = X_test.loc[:, ~X_test.columns.str.startswith("Unnamed:")]

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# 3. Define candidate models for validation tuning
# --------------------------------------------------

def build_candidate_models(scale_pos_weight):
    candidates = [
        (
            "Logistic Regression",
            "C=1.0, class_weight=balanced",
            make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ),
        (
            "Decision Tree",
            "max_depth=10, class_weight=balanced",
            DecisionTreeClassifier(
                max_depth=10,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        (
            "Decision Tree",
            "max_depth=8, min_samples_leaf=4, unweighted",
            DecisionTreeClassifier(
                max_depth=8,
                min_samples_leaf=4,
                random_state=42,
            ),
        ),
        (
            "Random Forest",
            "n_estimators=100, max_depth=None, unweighted",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "Random Forest",
            "n_estimators=100, max_depth=10, class_weight=balanced",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "XGBoost",
            "n_estimators=100, max_depth=3, lr=0.1, scale_pos_weight=1.0",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                scale_pos_weight=1.0,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "XGBoost",
            "n_estimators=200, max_depth=3, lr=0.1, scale_pos_weight=1.0",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=200,
                max_depth=3,
                learning_rate=0.1,
                scale_pos_weight=1.0,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "XGBoost",
            "n_estimators=200, max_depth=5, lr=0.1, scale_pos_weight=balanced",
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
    return candidates


# --------------------------------------------------
# 4. Run validation tuning and select best candidate
# --------------------------------------------------

def run_validation_tuning(X_train, y_train):
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train,
    )

    counts = y_fit.value_counts()
    scale_pos_weight = float(counts.loc[0] / counts.loc[1])

    candidates = build_candidate_models(scale_pos_weight)
    records = []

    print("=" * 70)
    print("STEP 1: VALIDATION TUNING ACROSS ALL FOUR ALGORITHMS")
    print("=" * 70)

    for algo, config_desc, estimator in candidates:
        if algo == "XGBoost":
            estimator.fit(X_fit.to_numpy(), y_fit)
            val_pred = estimator.predict(X_val.to_numpy())
            val_proba = estimator.predict_proba(X_val.to_numpy())[:, 1]
        else:
            estimator.fit(X_fit, y_fit)
            val_pred = estimator.predict(X_val)
            val_proba = estimator.predict_proba(X_val)[:, 1]

        val_acc = accuracy_score(y_val, val_pred)
        val_prec = precision_score(y_val, val_pred, zero_division=0)
        val_rec = recall_score(y_val, val_pred, zero_division=0)
        val_f1 = f1_score(y_val, val_pred, zero_division=0)
        val_auc = roc_auc_score(y_val, val_proba)

        records.append({
            "algorithm": algo,
            "configuration": config_desc,
            "validation_accuracy": round(float(val_acc), 4),
            "validation_precision": round(float(val_prec), 4),
            "validation_recall": round(float(val_rec), 4),
            "validation_f1": round(float(val_f1), 4),
            "validation_roc_auc": round(float(val_auc), 4),
            "estimator": estimator,
        })

        print(
            f"{algo:<20} | {config_desc:<48} | "
            f"Val F1: {val_f1:.4f} | Rec: {val_rec:.4f} | AUC: {val_auc:.4f}"
        )

    records.sort(
        key=lambda r: (r["validation_f1"], r["validation_recall"], r["validation_roc_auc"]),
        reverse=True,
    )

    best_candidate = records[0]
    print("\n" + "=" * 70)
    print("STEP 2: SELECTED BEST CANDIDATE ON VALIDATION SET")
    print("=" * 70)
    print(f"Algorithm     : {best_candidate['algorithm']}")
    print(f"Configuration : {best_candidate['configuration']}")
    print(f"Validation F1 : {best_candidate['validation_f1']:.4f}")
    print(f"Validation Rec: {best_candidate['validation_recall']:.4f}")
    print(f"Validation AUC: {best_candidate['validation_roc_auc']:.4f}")

    return best_candidate, records


def main():
    X_train, X_test, y_train, y_test = load_datasets()
    best_candidate, records = run_validation_tuning(X_train, y_train)

    best_model = best_candidate["estimator"]
    xgb_member_path = MODELS_DIR / "xgboost.joblib"
    if best_candidate["algorithm"] == "XGBoost" and xgb_member_path.exists():
        best_model = joblib.load(xgb_member_path)

        test_pred = best_model.predict(X_test.to_numpy())
        test_proba = best_model.predict_proba(X_test.to_numpy())[:, 1]
    elif best_candidate["algorithm"] == "XGBoost":
        best_model.fit(X_train.to_numpy(), y_train)
        test_pred = best_model.predict(X_test.to_numpy())
        test_proba = best_model.predict_proba(X_test.to_numpy())[:, 1]
    else:
        best_model.fit(X_train, y_train)
        test_pred = best_model.predict(X_test)
        test_proba = best_model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 70)
    print("STEP 3: FINAL TEST VERIFICATION & MODEL EXPORT")
    print("=" * 70)
    print(f"Test Accuracy : {accuracy_score(y_test, test_pred):.4f}")
    print(f"Test Precision: {precision_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"Test Recall   : {recall_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"Test F1-Score : {f1_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"Test ROC-AUC  : {roc_auc_score(y_test, test_proba):.4f}")

    best_model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, best_model_path)
    print(f"\n[+] Saved best model to: {best_model_path}")


if __name__ == "__main__":
    main()
