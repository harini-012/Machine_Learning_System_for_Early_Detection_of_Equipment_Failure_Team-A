
"""
Member 4: Automated Alert Management Tests

Tests:
1. Alert generation
2. Duplicate prevention
3. In-memory status updates
4. CSV status persistence
5. Invalid alert ID handling
6. Prevention of reopening resolved alerts
7. New alert generation after resolution
"""

import unittest
import tempfile
from pathlib import Path

import pandas as pd

try:
    from .alert_management import (
        generate_alerts,
        update_alert_status,
        update_alert_status_in_csv
    )
except ImportError:
    from alert_management import (
        generate_alerts,
        update_alert_status,
        update_alert_status_in_csv
    )


class TestAlertManagement(unittest.TestCase):

    def setUp(self):

        # Sample equipment records
        self.sample_data = pd.DataFrame([
            {
                "Equipment ID": "EQ001",
                "Failure Probability": 0.10,
                "Risk Level": "Healthy",
                "Abnormal Sensors": "None",
                "Risk Summary": "Healthy equipment condition."
            },
            {
                "Equipment ID": "EQ002",
                "Failure Probability": 0.45,
                "Risk Level": "Warning",
                "Abnormal Sensors": "Torque [Nm]",
                "Risk Summary": "Equipment requires monitoring."
            },
            {
                "Equipment ID": "EQ003",
                "Failure Probability": 0.85,
                "Risk Level": "Critical",
                "Abnormal Sensors": "Tool wear [min], Torque [Nm]",
                "Risk Summary": "Detailed equipment inspection recommended."
            }
        ])

        self.alerts = generate_alerts(self.sample_data)

        # Temporary CSV file.
        # The actual reports/generated_alerts.csv is not modified.
        self.temp_dir = tempfile.TemporaryDirectory()

        self.csv_file = (
            Path(self.temp_dir.name) / "test_alerts.csv"
        )

        pd.DataFrame(self.alerts).to_csv(
            self.csv_file,
            index=False
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_alert_generation(self):

        self.assertEqual(len(self.alerts), 2)

        risk_levels = {
            alert["risk_level"]
            for alert in self.alerts
        }

        self.assertEqual(
            risk_levels,
            {"Warning", "Critical"}
        )

        print("TEST 1 PASSED: Alert generation")

    def test_duplicate_prevention(self):

        duplicate_alerts = generate_alerts(
            self.sample_data,
            existing_alerts=self.alerts
        )

        self.assertEqual(len(duplicate_alerts), 0)

        print("TEST 2 PASSED: Duplicate prevention")

    def test_in_memory_status_updates(self):

        alert_id = self.alerts[0]["alert_id"]

        acknowledged = update_alert_status(
            self.alerts,
            alert_id,
            "Acknowledged"
        )

        self.assertEqual(
            acknowledged[0]["status"],
            "Acknowledged"
        )

        resolved = update_alert_status(
            acknowledged,
            alert_id,
            "Resolved"
        )

        self.assertEqual(
            resolved[0]["status"],
            "Resolved"
        )

        print("TEST 3 PASSED: In-memory status updates")

    def test_csv_status_persistence(self):

        alert_id = self.alerts[0]["alert_id"]

        # Update CSV status
        update_alert_status_in_csv(
            self.csv_file,
            alert_id,
            "Acknowledged"
        )

        # Read CSV again to verify saved status
        saved_alerts = pd.read_csv(self.csv_file)

        saved_status = saved_alerts.loc[
            saved_alerts["alert_id"] == alert_id,
            "status"
        ].iloc[0]

        self.assertEqual(
            saved_status,
            "Acknowledged"
        )

        print("TEST 4 PASSED: CSV status persistence")

    def test_invalid_alert_id(self):

        with self.assertRaises(ValueError):

            update_alert_status_in_csv(
                self.csv_file,
                "INVALID-ALERT-ID",
                "Acknowledged"
            )

        print("TEST 5 PASSED: Invalid alert ID handling")

    def test_resolved_alert_cannot_be_reopened(self):

        alert_id = self.alerts[0]["alert_id"]

        update_alert_status_in_csv(
            self.csv_file,
            alert_id,
            "Resolved"
        )

        with self.assertRaises(ValueError):

            update_alert_status_in_csv(
                self.csv_file,
                alert_id,
                "Acknowledged"
            )

        print("TEST 6 PASSED: Resolved alert cannot be reopened")

    def test_new_alert_after_resolution(self):

        alert_id = self.alerts[0]["alert_id"]

        resolved_alerts = update_alert_status(
            self.alerts,
            alert_id,
            "Resolved"
        )

        new_alerts = generate_alerts(
            self.sample_data,
            existing_alerts=resolved_alerts
        )

        self.assertEqual(len(new_alerts), 1)

        self.assertEqual(
            new_alerts[0]["equipment_id"],
            self.alerts[0]["equipment_id"]
        )

        print("TEST 7 PASSED: New alert after resolution")


if __name__ == "__main__":

    unittest.main(verbosity=2)
