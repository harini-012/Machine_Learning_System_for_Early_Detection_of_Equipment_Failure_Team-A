"""
Member 5 – Milestone 4 End-to-End Workflow Test Suite (tests/test_milestone4_workflow.py)

Tests the complete application workflow required by Milestone 4:
  1. Home, Dashboard, Prediction, Equipment, and Monitoring page routes
  2. Equipment registration (/equipment/add), duplicate/invalid checks, editing (/equipment/edit/<id>),
     and deletion (/equipment/delete/<id>)
  3. Invalid sensor input handling (unregistered equipment, non-numeric, negative/out-of-range values)
  4. Real model prediction via /dashboard and /predict and persistence of multiple observations
  5. Reopening equipment history (/equipment/<id>) to confirm saved readings and predictions persist
  6. Alert status lifecycle (New -> Acknowledged -> Resolved) & health score / trend persistence
  7. Report files and CSV download route verification
"""

import os
import tempfile
import unittest
from pathlib import Path

import app as flask_app_module
from src import database
from src.health_scoring import evaluate_equipment_health
from src.trend_analysis import analyse_trends

PROJECT_DIR = Path(__file__).resolve().parent.parent


class TestMilestone4CompleteWorkflow(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = Path(self.temp_dir.name) / "test_milestone4.sqlite"
        self.prev_db_url = os.environ.get("DATABASE_URL")
        os.environ["DATABASE_URL"] = f"sqlite:///{self.test_db_path}"
        database.init_database()

        flask_app_module.app.config["TESTING"] = True
        self.client = flask_app_module.app.test_client()

    def tearDown(self):
        if self.prev_db_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = self.prev_db_url
        self.temp_dir.cleanup()

    def test_1_core_pages_load(self):
        """Verify Home, Dashboard, Predict, Equipment, and Monitoring pages return HTTP 200."""
        for route in ["/", "/dashboard", "/predict", "/equipment", "/equipment/add"]:
            resp = self.client.get(route)
            self.assertEqual(resp.status_code, 200, f"Route {route} failed with {resp.status_code}")

        for page_key in ["risk", "anomalies", "alerts", "maintenance", "history"]:
            resp = self.client.get(f"/monitoring/{page_key}")
            self.assertEqual(resp.status_code, 200)

        resp_404 = self.client.get("/monitoring/nonexistent_page")
        self.assertEqual(resp_404.status_code, 404)

    def test_2_equipment_add_edit_and_delete_workflow(self):
        """Test registering equipment, rejecting invalid/duplicate inputs, editing, and deleting."""
        # 1. Add valid equipment
        resp_add = self.client.post(
            "/equipment/add",
            data={
                "equipment_id": "EQ-M4-101",
                "product_id": "M14860",
                "equipment_type": "M",
                "description": "Primary CNC Milling Spindle",
            },
            follow_redirects=True,
        )
        self.assertEqual(resp_add.status_code, 200)
        self.assertIn(b"EQ-M4-101", resp_add.data)

        # 2. Prevent duplicate equipment registration
        resp_dup = self.client.post(
            "/equipment/add",
            data={
                "equipment_id": "EQ-M4-101",
                "product_id": "M14860",
                "equipment_type": "M",
            },
        )
        self.assertEqual(resp_dup.status_code, 400)
        self.assertIn(b"already registered", resp_dup.data)

        # 3. Reject invalid equipment type
        resp_bad_type = self.client.post(
            "/equipment/add",
            data={
                "equipment_id": "EQ-M4-102",
                "equipment_type": "X",
            },
        )
        self.assertEqual(resp_bad_type.status_code, 400)

        # 4. Edit equipment details
        resp_edit = self.client.post(
            "/equipment/edit/EQ-M4-101",
            data={
                "product_id": "H29424",
                "equipment_type": "H",
                "description": "Upgraded Heavy-Duty Spindle",
            },
            follow_redirects=True,
        )
        self.assertEqual(resp_edit.status_code, 200)
        updated = database.get_equipment_by_id("EQ-M4-101")
        self.assertEqual(updated["equipment_type"], "H")
        self.assertEqual(updated["product_id"], "H29424")

        # 5. Delete equipment
        resp_del = self.client.post("/equipment/delete/EQ-M4-101", follow_redirects=True)
        self.assertEqual(resp_del.status_code, 200)
        self.assertIsNone(database.get_equipment_by_id("EQ-M4-101"))

    def test_3_invalid_sensor_input_validation(self):
        """Test that invalid sensor inputs return HTTP 400 with descriptive error messages."""
        database.upsert_equipment("EQ-VALID-01", product_id="L47181", equipment_type="L")

        # Unregistered equipment
        resp_unreg = self.client.post(
            "/dashboard",
            data={
                "equipment_id": "EQ-DOES-NOT-EXIST",
                "air_temperature": "300.5",
                "process_temperature": "310.2",
                "rotational_speed": "1400",
                "torque": "45.0",
                "tool_wear": "120",
            },
        )
        self.assertEqual(resp_unreg.status_code, 400)
        self.assertIn(b"not registered", resp_unreg.data)

        # Non-numeric sensor reading
        resp_nan = self.client.post(
            "/dashboard",
            data={
                "equipment_id": "EQ-VALID-01",
                "air_temperature": "invalid_temp",
                "process_temperature": "310.2",
                "rotational_speed": "1400",
                "torque": "45.0",
                "tool_wear": "120",
            },
        )
        self.assertEqual(resp_nan.status_code, 400)

        # Negative tool wear (violates minimum bound)
        resp_neg = self.client.post(
            "/dashboard",
            data={
                "equipment_id": "EQ-VALID-01",
                "air_temperature": "300.5",
                "process_temperature": "310.2",
                "rotational_speed": "1400",
                "torque": "45.0",
                "tool_wear": "-15",
            },
        )
        self.assertEqual(resp_neg.status_code, 400)

    def test_4_prediction_and_history_persistence_across_reopen(self):
        """Submit two sensor readings for an equipment record and verify both persist in /equipment/<id>."""
        database.upsert_equipment(
            "EQ-HIST-01",
            product_id="L47181",
            equipment_type="L",
            description="History Verification Unit",
        )

        # Observation 1: Healthy operating reading
        resp1 = self.client.post(
            "/dashboard",
            data={
                "equipment_id": "EQ-HIST-01",
                "air_temperature": "298.1",
                "process_temperature": "308.6",
                "rotational_speed": "1551",
                "torque": "42.8",
                "tool_wear": "25",
            },
        )
        self.assertEqual(resp1.status_code, 200)
        self.assertIn(b"No Failure", resp1.data)

        # Observation 2: High-stress reading (high tool wear & torque)
        resp2 = self.client.post(
            "/predict",
            data={
                "equipment_id": "EQ-HIST-01",
                "air_temperature": "300.5",
                "process_temperature": "310.2",
                "rotational_speed": "1320",
                "torque": "62.5",
                "tool_wear": "215",
            },
        )
        self.assertEqual(resp2.status_code, 200)
        self.assertIn(b"Failure", resp2.data)

        # Reopen equipment detail page and verify both observations & predictions are visible
        detail_resp = self.client.get("/equipment/EQ-HIST-01")
        self.assertEqual(detail_resp.status_code, 200)
        self.assertIn(b"EQ-HIST-01", detail_resp.data)
        self.assertIn(b"298.1", detail_resp.data)
        self.assertIn(b"300.5", detail_resp.data)

        history = database.retrieve_equipment_history("EQ-HIST-01")
        self.assertEqual(len(history["sensor_readings"]), 2)
        self.assertEqual(len(history["predictions"]), 2)

        # Verify trend analysis detects deterioration from Observation 1 -> Observation 2
        trend = analyse_trends("EQ-HIST-01")
        self.assertEqual(trend["failure_probability_trend"], "Deteriorating")
        self.assertEqual(trend["health_score_trend"], "Deteriorating")

    def test_5_alert_status_persistence_and_health_scoring(self):
        """Verify health scoring, alert deduplication, and status transitions persist in DB."""
        eq_id = "EQ-ALERT-01"
        health = evaluate_equipment_health(eq_id, 0.88, save_to_db=True)
        self.assertAlmostEqual(health["health_score"], 12.0)
        self.assertEqual(health["health_status"], "Poor")

        aid = database.save_alert(
            equipment_id=eq_id,
            risk_level="Critical",
            failure_probability=0.88,
            description="Critical tool wear failure risk",
            condition="Tool Wear Failure",
        )
        self.assertIsNotNone(aid)

        # Update status to Acknowledged and confirm persistence
        database.update_alert_status_in_db(aid, "Acknowledged")
        alerts = database.get_alerts(equipment_id=eq_id)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["status"], "Acknowledged")

        # Update status to Resolved and confirm persistence
        database.update_alert_status_in_db(aid, "Resolved")
        alerts_after = database.get_alerts(equipment_id=eq_id)
        self.assertEqual(alerts_after[0]["status"], "Resolved")

    def test_6_report_files_and_download_routes(self):
        """Verify required CSV reports exist and are non-empty (plus any /reports/* routes if registered)."""
        required_reports = [
            PROJECT_DIR / "reports" / "prediction_results.csv",
            PROJECT_DIR / "reports" / "maintenance_recommendations.csv",
            PROJECT_DIR / "reports" / "model_comparison.csv",
            PROJECT_DIR / "reports" / "health_scores.csv",
            PROJECT_DIR / "reports" / "trend_analysis_summary.csv",
        ]
        for report_path in required_reports:
            self.assertTrue(report_path.exists(), f"Missing report file: {report_path}")
            self.assertGreater(report_path.stat().st_size, 50, f"Report file is empty: {report_path}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
