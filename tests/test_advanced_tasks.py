"""
Automated Test Suite for Advanced Tasks T38, T39, T40, T41, and T42
(tests/test_advanced_tasks.py)

Verifies:
  - T38: Multiclass specific failure-type model (models/failure_type_model.joblib)
         and single-observation prediction (predict_specific_failure_type)
  - T39: Binary vs. Multiclass classification comparison report & CSV
  - T40: Isolation Forest anomaly detection investigation & Jupyter Notebook (.ipynb)
  - T41: Exact TreeSHAP & Permutation Feature Importance explainability report & local explanations
  - T42: Decision threshold & Precision-Recall comparison report & CSV
"""

import json
import unittest
from pathlib import Path

from src.failure_type_model import predict_specific_failure_type
from src.model_explainability import explain_single_prediction

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"


class TestAdvancedTasksT38ToT42(unittest.TestCase):
    def test_t38_failure_type_model_and_prediction(self):
        """Verify T38 failure-type model artifact, report CSV, and live failure-type inference."""
        self.assertTrue((MODELS_DIR / "failure_type_model.joblib").is_file())
        self.assertTrue((MODELS_DIR / "failure_type_classes.json").is_file())
        self.assertTrue((REPORTS_DIR / "t38_failure_type_model_report.csv").is_file())
        self.assertTrue((REPORTS_DIR / "t38_failure_type_confusion_matrix.png").is_file())

        pred = predict_specific_failure_type(
            {
                "Type": "L",
                "Air temperature [K]": 300.5,
                "Process temperature [K]": 310.2,
                "Rotational speed [rpm]": 1320.0,
                "Torque [Nm]": 62.5,
                "Tool wear [min]": 215.0,
            }
        )
        self.assertIn("predicted_failure_type", pred)
        self.assertIn("class_probabilities", pred)
        self.assertEqual(len(pred["class_probabilities"]), 6)

    def test_t39_binary_vs_multiclass_comparison_report(self):
        """Verify T39 binary vs. multiclass comparison report, metrics CSV, and chart."""
        self.assertTrue((REPORTS_DIR / "t39_binary_vs_multiclass_comparison.md").is_file())
        self.assertTrue((REPORTS_DIR / "t39_binary_vs_multiclass_metrics.csv").is_file())
        self.assertTrue((REPORTS_DIR / "t39_binary_vs_multiclass_chart.png").is_file())

    def test_t40_isolation_forest_anomaly_notebook(self):
        """Verify T40 Isolation Forest Jupyter Notebook, results CSV, and comparison chart."""
        nb_path = NOTEBOOKS_DIR / "T40_Isolation_Forest_Anomaly_Detection.ipynb"
        self.assertTrue(nb_path.is_file())
        nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb_data.get("nbformat"), 4)
        self.assertGreaterEqual(len(nb_data.get("cells", [])), 4)
        self.assertTrue((REPORTS_DIR / "t40_isolation_forest_results.csv").is_file())
        self.assertTrue((REPORTS_DIR / "t40_isolation_forest_comparison.png").is_file())

    def test_t41_explainability_shap_report_and_local_explanation(self):
        """Verify T41 Exact TreeSHAP explainability report, CSV, chart, and single-reading explanation."""
        self.assertTrue((REPORTS_DIR / "t41_explainability_report.md").is_file())
        self.assertTrue((REPORTS_DIR / "t41_feature_importance.csv").is_file())
        self.assertTrue((REPORTS_DIR / "t41_feature_importance_chart.png").is_file())

        drivers = explain_single_prediction(
            {
                "Type": "L",
                "Air temperature [K]": 300.5,
                "Process temperature [K]": 310.2,
                "Rotational speed [rpm]": 1320.0,
                "Torque [Nm]": 62.5,
                "Tool wear [min]": 215.0,
            },
            top_k=3,
        )
        self.assertEqual(len(drivers), 3)
        self.assertIn("feature", drivers[0])
        self.assertIn("shap_contribution", drivers[0])

    def test_t42_threshold_precision_recall_comparison(self):
        """Verify T42 decision threshold & Precision-Recall comparison report, CSV, and chart."""
        self.assertTrue((REPORTS_DIR / "t42_precision_recall_threshold_report.md").is_file())
        self.assertTrue((REPORTS_DIR / "t42_threshold_metrics.csv").is_file())
        self.assertTrue((REPORTS_DIR / "t42_precision_recall_threshold_chart.png").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
