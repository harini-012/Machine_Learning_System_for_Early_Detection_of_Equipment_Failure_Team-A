
"""
Member 4: Automated Alert Generation and Management

Consumes Member 2's anomaly_results.csv and generates
alerts based on equipment risk classification.

Integrates Member 3 maintenance recommendations.

Supports:
- Alert generation
- Duplicate prevention
- Alert status management
- Persistent CSV storage
"""

from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path
import math
import pandas as pd


ACTIVE_STATUSES = {"New", "Acknowledged"}

VALID_STATUSES = {
    "New",
    "Acknowledged",
    "Resolved"
}


def get_condition(abnormal_sensors):
    """Identify the condition associated with an alert."""

    if abnormal_sensors is None:
        return "Failure probability risk"

    try:
        if pd.isna(abnormal_sensors):
            return "Failure probability risk"
    except (TypeError, ValueError):
        pass

    condition = str(abnormal_sensors).strip()

    if condition.lower() in {"", "none", "nan", "null"}:
        return "Failure probability risk"

    return condition


def create_alert(row, recommendation=None):
    """Create an alert with failure and maintenance details."""

    recommendation = recommendation or {}

    equipment_id = str(row["Equipment ID"]).strip()
    risk_level = str(row["Risk Level"]).strip().capitalize()
    probability = float(row["Failure Probability"])

    if not equipment_id:
        raise ValueError("Equipment ID cannot be empty.")

    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(
            "Failure probability must be between 0 and 1."
        )

    if risk_level == "Healthy":
        return None

    if risk_level not in {"Warning", "Critical"}:
        raise ValueError(f"Invalid risk level: {risk_level}")

    # Get Member 3's primary failure issue.
    primary_issue = recommendation.get("primary_issue")

    if primary_issue is None or str(primary_issue).strip().lower() in {
        "", "none", "nan", "null"
    }:
        primary_issue = get_condition(
            row.get("Abnormal Sensors", "None")
        )

    # Get maintenance recommendations.
    actions = recommendation.get("recommended_actions", [])

    if isinstance(actions, str):
        actions = [actions] if actions.strip() else []

    if not isinstance(actions, (list, tuple)):
        actions = [str(actions)]

    actions = [
        str(action)
        for action in actions
        if action is not None and str(action).strip()
    ]

    action_type = recommendation.get(
        "action_type",
        "Inspection"
    )

    maintenance_summary = recommendation.get(
        "summary",
        row.get(
            "Risk Summary",
            "Equipment requires inspection."
        )
    )

    if not maintenance_summary:
        maintenance_summary = row.get(
            "Risk Summary",
            "Equipment requires inspection."
        )

    timestamp = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )

    description = (
        f"{risk_level} ALERT: Equipment {equipment_id}. "
        f"Failure probability: {probability:.2%}. "
        f"Possible issue: {primary_issue}. "
        f"{maintenance_summary}. "
        f"Recommended maintenance: "
        f"{'; '.join(actions) if actions else 'Detailed inspection recommended.'}"
    )

    return {
        "alert_id": str(uuid4()),
        "equipment_id": equipment_id,
        "timestamp": timestamp,
        "failure_probability": probability,
        "risk_level": risk_level,
        "condition": str(primary_issue),
        "primary_issue": str(primary_issue),
        "action_type": str(action_type),
        "recommended_actions": "; ".join(actions),
        "estimated_downtime": recommendation.get(
            "estimated_downtime",
            "Not specified"
        ),
        "alert_description": description,
        "status": "New"
    }


def is_duplicate(alert, existing_alerts):
    """
    Prevent duplicate active alerts for the same equipment
    and condition.
    """

    for existing in existing_alerts:

        same_equipment = (
            str(existing.get("equipment_id", "")).strip()
            == alert["equipment_id"]
        )

        same_condition = (
            str(existing.get("condition", "")).strip().lower()
            == alert["condition"].lower()
        )

        active = existing.get("status") in ACTIVE_STATUSES

        if same_equipment and same_condition and active:
            return True

    return False


def generate_alerts(
    anomaly_results,
    existing_alerts=None,
    recommendations_by_equipment=None
):
    """Generate alerts using Member 2 and Member 3 results."""

    if existing_alerts is None:
        existing_alerts = []

    if recommendations_by_equipment is None:
        recommendations_by_equipment = {}

    required_columns = {
        "Equipment ID",
        "Failure Probability",
        "Risk Level",
        "Abnormal Sensors",
        "Risk Summary"
    }

    missing = required_columns - set(anomaly_results.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    alerts = []
    known_alerts = list(existing_alerts)

    for _, row in anomaly_results.iterrows():

        equipment_id = str(row["Equipment ID"]).strip()

        recommendation = recommendations_by_equipment.get(
            equipment_id,
            {}
        )

        alert = create_alert(row, recommendation)

        if alert is None:
            continue

        if is_duplicate(alert, known_alerts):
            continue

        alerts.append(alert)
        known_alerts.append(alert)

    return alerts


def update_alert_status(alerts, alert_id, new_status):
    """Update an alert status in an in-memory alert list."""

    if new_status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status. Choose from {sorted(VALID_STATUSES)}"
        )

    updated_alerts = []
    found = False

    for alert in alerts:

        updated = alert.copy()

        if updated["alert_id"] == alert_id:

            if updated["status"] == "Resolved":
                raise ValueError(
                    "Resolved alerts cannot be reopened."
                )

            updated["status"] = new_status
            found = True

        updated_alerts.append(updated)

    if not found:
        raise ValueError(
            f"Alert ID not found: {alert_id}"
        )

    return updated_alerts


def update_alert_status_in_csv(
    alert_file,
    alert_id,
    new_status
):
    """
    Permanently update an alert status in the CSV file.

    Supported statuses:
    New, Acknowledged, Resolved
    """

    alert_file = Path(alert_file)

    if not alert_file.exists():
        raise FileNotFoundError(
            f"Alert file not found: {alert_file}"
        )

    if new_status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status. Choose from {sorted(VALID_STATUSES)}"
        )

    df = pd.read_csv(alert_file, dtype=str).fillna("")

    if "alert_id" not in df.columns or "status" not in df.columns:
        raise ValueError(
            "CSV must contain alert_id and status columns."
        )

    matching = (
        df["alert_id"].astype(str) == str(alert_id)
    )

    if not matching.any():
        raise ValueError(
            f"Alert ID not found: {alert_id}"
        )

    current_status = str(
        df.loc[matching, "status"].iloc[0]
    )

    if current_status == "Resolved":
        raise ValueError(
            "Resolved alerts cannot be reopened."
        )

    df.loc[matching, "status"] = new_status

    df.to_csv(alert_file, index=False)

    print(
        f"Alert {alert_id} status updated: "
        f"{current_status} -> {new_status}"
    )

    return df.to_dict("records")
