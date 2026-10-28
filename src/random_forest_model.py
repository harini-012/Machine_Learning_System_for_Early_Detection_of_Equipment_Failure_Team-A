from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score,)

# Setup paths relative to script location
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_member1_datasets():
    """Loads the pre-split feature and target datasets prepared by Member 1."""
    x_train_path = DATA_DIR / "X_train.csv"
    x_test_path = DATA_DIR / "X_test.csv"
    y_train_path = DATA_DIR / "y_train.csv"
    y_test_path = DATA_DIR / "y_test.csv"

    for p in [x_train_path, x_test_path, y_train_path, y_test_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Required dataset not found: {p}\n"
                "Please verify Member 1 has exported the split CSVs."
            )

    print("=" * 65)
    print("STEP 1: LOADING PREPARED DATASETS")
    print("=" * 65)

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    y_test = pd.read_csv(y_test_path).values.ravel()

    print(f"[+] Features loaded ({X_train.shape[1]} columns): {list(X_train.columns)}")
    print(f"[+] X_train shape: {X_train.shape} | X_test shape: {X_test.shape}")
    print(f"[+] y_train distribution -> Normal: {np.sum(y_train == 0)}, Failure: {np.sum(y_train == 1)}")
    print(f"[+] y_test  distribution -> Normal: {np.sum(y_test == 0)}, Failure: {np.sum(y_test == 1)}\n")

    return X_train, X_test, y_train, y_test


def evaluate_configuration(name, model, X_train, y_train, X_test, y_test):
    """Fits and evaluates a model instance, returning a summary dictionary."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    return {
        "Config": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "model_obj": model,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }


def run_random_forest_experiments(X_train, X_test, y_train, y_test):
    """Tests configurations across tree depth, tree counts, and class weighting."""
    print("=" * 65)
    print("STEP 2: HYPERPARAMETER & CLASS WEIGHTING EXPERIMENTS")
    print("=" * 65)

    experiments = [
        (
            "Default (n=100, depth=None, unweighted)",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "Balanced Weights (n=100, depth=10, balanced)",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "Tuned Model (n=200, depth=12, balanced_subsample, min_leaf=2)",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_split=4,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    results = []
    for name, clf in experiments:
        res = evaluate_configuration(name, clf, X_train, y_train, X_test, y_test)
        results.append(res)
        print(
            f"[*] {name[:35]:<35} | Recall: {res['Recall']:.4f} | "
            f"Precision: {res['Precision']:.4f} | F1: {res['F1-Score']:.4f} | ROC-AUC: {res['ROC-AUC']:.4f}"
        )

    # Prioritize F1-Score while maintaining high Recall (crucial for maintenance failure alerting)
    best_exp = max(results, key=lambda x: (x["F1-Score"], x["Recall"]))
    print(f"\n[+] Selected Configuration: {best_exp['Config']}")

    return best_exp


def finalize_deliverables(best_result, X_train, X_test, y_test):
    """Outputs evaluation matrices, feature importance rankings, and saved artifact."""
    best_model = best_result["model_obj"]
    y_pred = best_result["y_pred"]

    print("\n" + "=" * 65)
    print("STEP 3: DETAILED EVALUATION & ARTIFACT EXPORT")
    print("=" * 65)

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  True Negatives (Correct Normal) : {cm[0][0]}")
    print(f"  False Positives (False Alarm)   : {cm[0][1]}")
    print(f"  False Negatives (Missed Failure): {cm[1][0]}  <-- Keep as low as possible")
    print(f"  True Positives (Detected Failure): {cm[1][1]}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    # 1. Feature Importance extraction
    importances = best_model.feature_importances_
    features = list(X_train.columns)
    fi_df = pd.DataFrame(
        {"Feature": features, "Importance": importances}
    ).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    fi_path = REPORTS_DIR / "feature_importance.csv"
    fi_df.to_csv(fi_path, index=False)
    print("Feature Importance Ranking:")
    print(fi_df.to_string(index=True))
    print(f"\n[✓] Saved feature importance rankings to: {fi_path}")

    # 2. Model serialization
    model_path = MODELS_DIR / "random_forest.joblib"
    joblib.dump(best_model, model_path)
    print(f"[✓] Saved trained Random Forest model to : {model_path}")
    print("=" * 65)


def main():
    X_train, X_test, y_train, y_test = load_member1_datasets()
    best_result = run_random_forest_experiments(X_train, X_test, y_train, y_test)
    finalize_deliverables(best_result, X_train, X_test, y_test)


if __name__ == "__main__":
    main()