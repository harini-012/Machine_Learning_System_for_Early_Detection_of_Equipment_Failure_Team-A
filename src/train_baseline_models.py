from pathlib import Path

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load Member 1's datasets
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Define the two baseline models
models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        class_weight="balanced",
        random_state=42
    ),
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
    )
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Train model
    model.fit(X_train, y_train)

    # Predict test labels
    predictions = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    # Save model
    filename = name.lower().replace(" ", "_")
    joblib.dump(model, MODELS_DIR / f"{filename}.joblib")

    # Save confusion matrix image
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["No Failure", "Failure"],
        values_format="d"
    )
    display.ax_.set_title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        REPORTS_DIR / f"{filename}_confusion_matrix.png",
        dpi=150
    )
    plt.close()

    # Collect metrics
    results.append({
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

# Save results for Member 5
results_df = pd.DataFrame(results)
results_df.to_csv(
    REPORTS_DIR / "baseline_results.csv",
    index=False
)

print("\n--- COMPLETED SUCCESSFULLY ---")
print(results_df.to_string(index=False))
print("\nModels saved in:", MODELS_DIR)
print("Reports saved in:", REPORTS_DIR)