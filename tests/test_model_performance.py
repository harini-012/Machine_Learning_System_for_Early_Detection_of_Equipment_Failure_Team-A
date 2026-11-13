"""
Member 5 – Held-Out Test Set Model Performance Verification (tests/test_model_performance.py)

Verifies `models/best_model.joblib` and `src/prediction_service.py` on the
2,000-record held-out test set (`data/processed/X_test.csv` & `y_test.csv`)
and records Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix
to `reports/heldout_test_verification.json` and `reports/heldout_test_verification.csv`.
"""

import json
import unittest
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.model_integration import load_best_model, predict_failure
from src.prediction_service import create_prediction_service

PROJECT_DIR = Path(__file__).resolve().parent.parent
X_TEST_PATH = PROJECT_DIR / "data" / "processed" / "X_test.csv"
Y_TEST_PATH = PROJECT_DIR / "data" / "processed" / "y_test.csv"
JSON_OUT = PROJECT_DIR / "reports" / "heldout_test_verification.json"
CSV_OUT = PROJECT_DIR / "reports" / "heldout_test_verification.csv"


def evaluate_heldout_test_set() -> dict:
    """Run held-out evaluation on X_test.csv and y_test.csv and save verification reports."""
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).iloc[:, 0].astype(int).to_numpy()

    model = load_best_model()
    results_df = predict_failure(X_test, model=model)

    y_pred = results_df["predicted_failure"].astype(int).to_numpy()
    y_prob = results_df["failure_probability"].astype(float).to_numpy()

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y_test, y_pred).ravel()]

    metrics = {
        "model_name": type(model).__name__,
        "test_records": int(len(X_test)),
        "actual_failures": int(y_test.sum()),
        "predicted_failures": int(y_pred.sum()),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(CSV_OUT, index=False)
    return metrics


class TestHeldOutModelPerformance(unittest.TestCase):
    def test_1_heldout_metrics_meet_quality_thresholds(self):
        """Verify held-out test set metrics on 2,000 test records and save results."""
        metrics = evaluate_heldout_test_set()

        self.assertEqual(metrics["test_records"], 2000)
        self.assertGreaterEqual(metrics["accuracy"], 0.98)
        self.assertGreaterEqual(metrics["precision"], 0.80)
        self.assertGreaterEqual(metrics["recall"], 0.65)
        self.assertGreaterEqual(metrics["f1_score"], 0.75)
        self.assertGreaterEqual(metrics["roc_auc"], 0.95)
        self.assertTrue(JSON_OUT.exists())
        self.assertTrue(CSV_OUT.exists())

    def test_2_single_reading_prediction_service_consistency(self):
        """Confirm Flask prediction_service matches model_integration on sample inputs."""
        predict_reading = create_prediction_service()
        sample = {
            "equipment_id": "EQ-VERIFY-01",
            "Type": "L",
            "Air temperature [K]": 300.5,
            "Process temperature [K]": 310.2,
            "Rotational speed [rpm]": 1320.0,
            "Torque [Nm]": 62.5,
            "Tool wear [min]": 215.0,
        }
        service_res = predict_reading(sample)
        batch_res = predict_failure(sample).iloc[0]

        self.assertEqual(service_res["equipment_id"], "EQ-VERIFY-01")
        self.assertEqual(service_res["predicted_failure"], int(batch_res["predicted_failure"]))
        self.assertAlmostEqual(
            round(service_res["failure_probability"], 4),
            float(batch_res["failure_probability"]),
            places=3,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
