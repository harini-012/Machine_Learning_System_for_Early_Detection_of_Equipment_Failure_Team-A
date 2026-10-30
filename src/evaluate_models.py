from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from train_baseline_models import X_test

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
# 2. Load prepared datasets from Member 1
# --------------------------------------------------

def load_datasets():
    required_files = {
        "X_train": DATA_DIR / "X_train.csv",
        "X_test": DATA_DIR / "X_test.csv",
        "y_train": DATA_DIR / "y_train.csv",
        "y_test": DATA_DIR / "y_test.csv",
    }

    for name, path in required_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Required dataset not found: {path}")

    X_train = pd.read_csv(required_files["X_train"])
    X_test = pd.read_csv(required_files["X_test"])
    y_train = pd.read_csv(required_files["y_train"]).iloc[:, 0].astype(int)
    y_test = pd.read_csv(required_files["y_test"]).iloc[:, 0].astype(int)

    X_train = X_train.loc[:, ~X_train.columns.str.startswith("Unnamed:")]
    X_test = X_test.loc[:, ~X_test.columns.str.startswith("Unnamed:")]

    print("=" * 65)
    print("STEP 1: LOADED DATASETS FOR COMMON EVALUATION")
    print("=" * 65)
    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape : {X_test.shape}")
    print(f"Training failures      : {int(y_train.sum())} / {len(y_train)}")
    print(f"Testing failures       : {int(y_test.sum())} / {len(y_test)}\n")

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# 3. Collect member reports and load trained models
# --------------------------------------------------

def collect_member_artifacts():
    print("=" * 65)
    print("STEP 2: COLLECTING ARTIFACTS FROM MEMBERS 2, 3, AND 4")
    print("=" * 65)

    baseline_path = REPORTS_DIR / "baseline_results.csv"
    rf_fi_path = REPORTS_DIR / "feature_importance.csv"
    xgb_metrics_path = REPORTS_DIR / "xgboost_metrics.json"

    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
        print("[Member 2] Baseline results loaded from reports/baseline_results.csv:")
        print(baseline_df.to_string(index=False))
        print()

    if rf_fi_path.exists():
        rf_fi = pd.read_csv(rf_fi_path)
        print("[Member 3] Random Forest top 3 features from reports/feature_importance.csv:")
        print(rf_fi.head(3).to_string(index=False))
        print()

    if xgb_metrics_path.exists():
        with open(xgb_metrics_path, "r", encoding="utf-8") as f:
            xgb_metrics = json.load(f)
        print("[Member 4] XGBoost metrics loaded from reports/xgboost_metrics.json:")
        print(
            f"  Accuracy: {xgb_metrics['accuracy']:.4f}, "
            f"Precision: {xgb_metrics['precision']:.4f}, "
            f"Recall: {xgb_metrics['recall']:.4f}, "
            f"F1: {xgb_metrics['f1_score']:.4f}, "
            f"ROC-AUC: {xgb_metrics['roc_auc']:.4f}\n"
        )


def load_models():
    model_files = {
        "Logistic Regression": MODELS_DIR / "logistic_regression.joblib",
        "Decision Tree": MODELS_DIR / "decision_tree.joblib",
        "Random Forest": MODELS_DIR / "random_forest.joblib",
        "XGBoost": MODELS_DIR / "xgboost.joblib",
    }

    models = {}
    for name, path in model_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Model file for {name} not found at: {path}")
        models[name] = joblib.load(path)

    return models


# --------------------------------------------------
# 4. Evaluate all models on validation & test sets
# --------------------------------------------------

def evaluate_all_models(models, X_train, X_test, y_train, y_test):
    print("=" * 65)
    print("STEP 3: UNIFIED VALIDATION & TEST EVALUATION")
    print("=" * 65)

    # Stratified validation split from training data for fair model selection
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train,
    )

    comparison_rows = []
    confusion_matrices = {}

    for name, trained_model in models.items():
        # Validation evaluation using a cloned estimator fit on X_fit
        val_model = clone(trained_model)
        if name == "XGBoost":
            val_model.fit(X_fit.to_numpy(), y_fit)
            val_pred = val_model.predict(X_val.to_numpy())
            val_proba = val_model.predict_proba(X_val.to_numpy())[:, 1]
        else:
            val_model.fit(X_fit, y_fit)
            val_pred = val_model.predict(X_val)
            val_proba = val_model.predict_proba(X_val)[:, 1]

        val_acc = accuracy_score(y_val, val_pred)
        val_prec = precision_score(y_val, val_pred, zero_division=0)
        val_rec = recall_score(y_val, val_pred, zero_division=0)
        val_f1 = f1_score(y_val, val_pred, zero_division=0)
        val_auc = roc_auc_score(y_val, val_proba)

        # Test evaluation using the full-training-set model
        if name == "XGBoost":
            test_pred = trained_model.predict(X_test.to_numpy())
            test_proba = trained_model.predict_proba(X_test.to_numpy())[:, 1]
        else:
            test_pred = trained_model.predict(X_test)
            test_proba = trained_model.predict_proba(X_test)[:, 1]

        test_acc = accuracy_score(y_test, test_pred)
        test_prec = precision_score(y_test, test_pred, zero_division=0)
        test_rec = recall_score(y_test, test_pred, zero_division=0)
        test_f1 = f1_score(y_test, test_pred, zero_division=0)
        test_auc = roc_auc_score(y_test, test_proba)

        cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        confusion_matrices[name] = cm

        # Save individual confusion matrix plot if not already exported by team member
        slug = name.lower().replace(" ", "_")
        cm_path = REPORTS_DIR / f"{slug}_confusion_matrix.png"
        if not cm_path.exists():
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["No Failure", "Failure"],
            )
            disp.plot(values_format="d", cmap="Blues")
            plt.title(f"{name} Confusion Matrix")
            plt.tight_layout()
            plt.savefig(cm_path, dpi=150)
            plt.close()

        row = {
            "model": name,
            "validation_accuracy": round(float(val_acc), 4),
            "validation_precision": round(float(val_prec), 4),
            "validation_recall": round(float(val_rec), 4),
            "validation_f1": round(float(val_f1), 4),
            "validation_roc_auc": round(float(val_auc), 4),
            "accuracy": round(float(test_acc), 4),
            "precision": round(float(test_prec), 4),
            "recall": round(float(test_rec), 4),
            "f1_score": round(float(test_f1), 4),
            "roc_auc": round(float(test_auc), 4),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }
        comparison_rows.append(row)

        print(
            f"{name:<20} | Val F1: {val_f1:.4f} (Rec: {val_rec:.4f}) | "
            f"Test F1: {test_f1:.4f} | Prec: {test_prec:.4f} | "
            f"Rec: {test_rec:.4f} | ROC-AUC: {test_auc:.4f}"
        )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by=["validation_f1", "validation_roc_auc"],
        ascending=False,
    ).reset_index(drop=True)

    return comparison_df, confusion_matrices


# --------------------------------------------------
# 5. Save comparison plots and select best model
# --------------------------------------------------

def save_comparison_visualizations(comparison_df, confusion_matrices):
    # Combined 2x2 confusion matrix grid
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()

    for idx, model_name in enumerate(comparison_df["model"]):
        cm = confusion_matrices[model_name]
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No Failure", "Failure"],
        )
        disp.plot(ax=axes[idx], values_format="d", cmap="Blues", colorbar=False)
        axes[idx].set_title(f"{model_name} Confusion Matrix", fontweight="bold")

    plt.tight_layout()
    combined_cm_path = REPORTS_DIR / "all_models_confusion_matrices.png"
    plt.savefig(combined_cm_path, dpi=150)
    plt.close()

    # Metric comparison bar chart
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    x = np.arange(len(comparison_df["model"]))
    width = 0.15

    plt.figure(figsize=(11, 6))
    for i, metric in enumerate(metrics_to_plot):
        plt.bar(
            x + (i - 2) * width,
            comparison_df[metric],
            width=width,
            label=metric.replace("_", " ").upper(),
        )

    plt.xticks(x, comparison_df["model"])
    plt.ylim(0, 1.08)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison on Test Set")
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    chart_path = REPORTS_DIR / "model_comparison_chart.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()


def main():
    X_train, X_test, y_train, y_test = load_datasets()
    collect_member_artifacts()
    models = load_models()

    comparison_df, confusion_matrices = evaluate_all_models(
        models, X_train, X_test, y_train, y_test
    )

    comparison_path = REPORTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    save_comparison_visualizations(comparison_df, confusion_matrices)

    best_row = comparison_df.iloc[0]
    best_model_name = best_row["model"]
    best_model = models[best_model_name]

    best_model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(best_model, best_model_path)

    print("\n" + "=" * 65)
    print("STEP 4: MODEL SELECTION & DELIVERABLES SAVED")
    print("=" * 65)
    print("\nModel Comparison Table:")
    display_cols = [
        "model",
        "validation_f1",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    ]
    print(comparison_df[display_cols].to_string(index=False))
    print(f"\n[+] Best Candidate Selected (by Validation F1): {best_model_name}")
    print(f"[+] Saved model comparison table to : {comparison_path}")
    print(f"[+] Saved best model artifact to    : {best_model_path}")
    print(f"[+] Saved confusion matrices to     : {REPORTS_DIR}")


if __name__ == "__main__":
    main()
