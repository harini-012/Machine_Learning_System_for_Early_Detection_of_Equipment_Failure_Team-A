"""
Automated Test Suite for Advanced Tasks T38 through T45
(tests/test_advanced_tasks.py)

Verifies:
  - T38: Multiclass specific failure-type model (models/failure_type_model.joblib)
         and single-observation prediction (predict_specific_failure_type)
  - T39: Binary vs. Multiclass classification comparison report & CSV
  - T40: Isolation Forest anomaly detection investigation & Jupyter Notebook (.ipynb)
  - T41: Exact TreeSHAP & Permutation Feature Importance explainability report & local explanations
  - T42: Decision threshold & Precision-Recall comparison report & CSV
  - T43: Cross-dataset model generalization test report, external telemetry CSV, and chart
  - T44: Simulated sensor-monitoring pipeline architecture diagram, markdown spec, and 20-step stream log
  - T45: Multi-channel maintenance alert notification design, deduplication logic, and dispatch demo
"""

import json
import unittest
from pathlib import Path

from src.failure_type_model import predict_specific_failure_type
from src.model_explainability import explain_single_prediction
from src.maintenance_alert_notifier import dispatch_maintenance_alert

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
NOTEBOOKS_DIR = PROJECT_DIR / "notebooks"


class TestAdvancedTasksT38ToT45(unittest.TestCase):
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

    def test_t43_cross_dataset_generalization_report(self):
        """Verify T43 cross-dataset generalization report, external telemetry CSV, and chart."""
        self.assertTrue((PROJECT_DIR / "data" / "external" / "external_plant_telemetry.csv").is_file())
        self.assertTrue((REPORTS_DIR / "t43_generalization_report.md").is_file())
        self.assertTrue((REPORTS_DIR / "t43_generalization_metrics.csv").is_file())
        self.assertTrue((REPORTS_DIR / "t43_generalization_chart.png").is_file())

    def test_t44_simulated_sensor_monitoring_pipeline(self):
        """Verify T44 pipeline architecture diagram, markdown spec, and 20-step stream log."""
        self.assertTrue((REPORTS_DIR / "t44_pipeline_architecture_diagram.png").is_file())
        self.assertTrue((REPORTS_DIR / "t44_pipeline_architecture_and_prototype.md").is_file())
        self.assertTrue((REPORTS_DIR / "t44_simulated_stream_log.csv").is_file())

    def test_t45_maintenance_alert_design_and_demo(self):
        """Verify T45 multi-channel alert dispatch design, deduplication, escalation, and logs."""
        self.assertTrue((REPORTS_DIR / "t45_alert_design_and_demo.md").is_file())
        self.assertTrue((REPORTS_DIR / "t45_alert_dispatch_log.json").is_file())
        self.assertTrue((REPORTS_DIR / "t45_alert_dispatch_log.csv").is_file())

        open_state = {}
        ev_warn = {
            "equipment_id": "EQ-TEST-45",
            "risk_level": "Warning",
            "failure_probability": 0.45,
            "primary_issue": "Tool Wear Failure (TWF)",
        }
        r1 = dispatch_maintenance_alert(ev_warn, active_open_alerts=open_state)
        self.assertTrue(r1["dispatched"])
        self.assertIn("SMTP_EMAIL", r1["channels_triggered"])

        # Duplicate warning should be suppressed
        r2 = dispatch_maintenance_alert(ev_warn, active_open_alerts=open_state)
        self.assertFalse(r2["dispatched"])

        # Escalation to Critical should override and trigger PagerDuty/SMS too
        ev_crit = {**ev_warn, "risk_level": "Critical", "failure_probability": 0.92}
        r3 = dispatch_maintenance_alert(ev_crit, active_open_alerts=open_state)
        self.assertTrue(r3["dispatched"])
        self.assertIn("PAGERDUTY_SMS", r3["channels_triggered"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
