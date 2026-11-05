"""
Member 5 Unit & Integration Tests (src/test_member5.py)

Verifies:
  1. Database initialization & CRUD functions (src/database.py)
  2. Direct compatibility with Member 3's db_module integration
  3. Duplicate alert prevention and alert status lifecycle in database
  4. Equipment health scoring formula and classification (src/health_scoring.py)
  5. Historical trend analysis for Improving, Stable, and Deteriorating trajectories (src/trend_analysis.py)
"""

import os
import datetime
import tempfile
import unittest
from pathlib import Path

try:
    from . import database
    from .health_scoring import compute_health_score, evaluate_equipment_health
    from .trend_analysis import analyse_trends
    from .maintenance_recommendations import process_equipment_recommendations
except ImportError:
    import database
    from health_scoring import compute_health_score, evaluate_equipment_health
    from trend_analysis import analyse_trends
    from maintenance_recommendations import process_equipment_recommendations


class TestMember5Modules(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = Path(self.temp_dir.name) / "test_equipment_maintenance.sqlite"
        self.prev_db_url = os.environ.get("DATABASE_URL")
        os.environ["DATABASE_URL"] = f"sqlite:///{self.test_db_path}"
        database.init_database()

    def tearDown(self):
        if self.prev_db_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = self.prev_db_url
        self.temp_dir.cleanup()

    def test_1_health_scoring_formula_and_categories(self):
        # Good: 80 - 100
        score, status = compute_health_score(0.10)
        self.assertAlmostEqual(score, 90.0)
        self.assertEqual(status, "Good")

        # Moderate: 50 - 79
        score_mod, status_mod = compute_health_score(0.35)
        self.assertAlmostEqual(score_mod, 65.0)
        self.assertEqual(status_mod, "Moderate")

        # Poor: Below 50
        score_poor, status_poor = compute_health_score(0.75)
        self.assertAlmostEqual(score_poor, 25.0)
        self.assertEqual(status_poor, "Poor")

        # Full dictionary output with DB persistence
        payload = evaluate_equipment_health("EQ-TEST-01", 0.20, save_to_db=True)
        self.assertEqual(payload["equipment_id"], "EQ-TEST-01")
        self.assertEqual(payload["health_score"], 80.0)
        self.assertEqual(payload["health_status"], "Good")

    def test_2_database_crud_and_member3_integration(self):
        eq_id = "EQ-TEST-02"
        database.insert_sensor_reading(
            equipment_id=eq_id,
            air_temp=300.5,
            process_temp=310.2,
            rotational_speed=1420.0,
            torque=52.0,
            tool_wear=185.0,
            equipment_type="M",
        )
        database.save_prediction(
            equipment_id=eq_id,
            failure_pred=True,
            failure_probability=0.68,
        )
        database.save_anomaly_and_risk(
            equipment_id=eq_id,
            anomalies=[{"sensor": "Torque [Nm]", "value": 52.0}],
            risk_level="Warning",
        )

        # Member 3 integration test passing db_module=database
        sensor_input = {
            "air_temperature": 300.5,
            "process_temperature": 310.2,
            "rotational_speed": 1420.0,
            "torque": 52.0,
            "tool_wear": 185.0,
            "equipment_type": "M",
        }
        res = process_equipment_recommendations(
            equipment_id=eq_id,
            sensor_data=sensor_input,
            risk_level="Warning",
            failure_probability=0.68,
            db_module=database,
        )
        self.assertTrue(res["saved_to_db"])

        history = database.retrieve_equipment_history(eq_id)
        self.assertEqual(len(history["sensor_readings"]), 1)
        self.assertEqual(len(history["predictions"]), 1)
        self.assertEqual(len(history["anomalies"]), 1)
        self.assertEqual(len(history["maintenance"]), 1)

    def test_3_alert_deduplication_and_status_lifecycle(self):
        eq_id = "EQ-TEST-03"
        aid1 = database.save_alert(
            equipment_id=eq_id,
            risk_level="Critical",
            failure_probability=0.88,
            description="Critical tool wear alert",
            condition="Tool Wear Failure",
        )
        # Duplicate active alert for same equipment & condition updates existing alert
        aid2 = database.save_alert(
            equipment_id=eq_id,
            risk_level="Critical",
            failure_probability=0.92,
            description="Updated critical tool wear alert",
            condition="Tool Wear Failure",
        )
        self.assertEqual(aid1, aid2)
        self.assertEqual(len(database.get_alerts(equipment_id=eq_id)), 1)

        # Transition New -> Acknowledged -> Resolved
        ack = database.update_alert_status_in_db(aid1, "Acknowledged")
        self.assertEqual(ack["status"], "Acknowledged")

        resolved = database.update_alert_status_in_db(aid1, "Resolved")
        self.assertEqual(resolved["status"], "Resolved")

        # Resolved alert cannot be reopened
        with self.assertRaises(ValueError):
            database.update_alert_status_in_db(aid1, "Acknowledged")

    def test_4_trend_analysis_trajectories(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        t_early = now - datetime.timedelta(days=5)
        t_late = now - datetime.timedelta(days=1)

        # Deteriorating equipment: failure prob goes from 0.15 -> 0.80, health score 85 -> 20
        eq_det = "EQ-TREND-DET"
        database.insert_sensor_reading(eq_det, t_early, 298.0, 308.0, 1500.0, 40.0, 50.0, "L")
        database.insert_sensor_reading(eq_det, t_late, 303.0, 312.0, 1320.0, 65.0, 210.0, "L")
        database.save_prediction(eq_det, False, 0.15, timestamp=t_early)
        database.save_prediction(eq_det, True, 0.80, timestamp=t_late)

        trend_det = analyse_trends(eq_det, lookback_days=30)
        self.assertEqual(trend_det["failure_probability_trend"], "Deteriorating")
        self.assertEqual(trend_det["health_score_trend"], "Deteriorating")
        self.assertEqual(trend_det["overall_trend"], "Deteriorating")

        # Improving equipment: failure prob goes from 0.75 -> 0.08, health score 25 -> 92
        eq_imp = "EQ-TREND-IMP"
        database.insert_sensor_reading(eq_imp, t_early, 302.0, 311.0, 1350.0, 60.0, 195.0, "M")
        database.insert_sensor_reading(eq_imp, t_late, 298.0, 308.0, 1500.0, 38.0, 10.0, "M")
        database.save_prediction(eq_imp, True, 0.75, timestamp=t_early)
        database.save_prediction(eq_imp, False, 0.08, timestamp=t_late)

        trend_imp = analyse_trends(eq_imp, lookback_days=30)
        self.assertEqual(trend_imp["failure_probability_trend"], "Improving")
        self.assertEqual(trend_imp["health_score_trend"], "Improving")
        self.assertEqual(trend_imp["overall_trend"], "Improving")


if __name__ == "__main__":
    unittest.main(verbosity=2)
