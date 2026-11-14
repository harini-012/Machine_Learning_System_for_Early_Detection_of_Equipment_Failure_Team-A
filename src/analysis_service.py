"""
Milestone 4 – Post-Prediction Analysis & Monitoring Service (src/analysis_service.py)

Integrates Milestone 3 modules after each Flask prediction:
  1. Equipment Health Score & Status (src/health_scoring.py)
  2. Sensor Anomaly Detection & Risk Classification (IQR Operating Thresholds)
  3. AI4I Failure Category Identification & Maintenance Recommendations (src/maintenance_recommendations.py)
  4. Automated Alert Generation & Deduplication (src/alert_management.py)
  5. Shared SQL Database Persistence & Historical Trend Analysis (src/database.py, src/trend_analysis.py)
"""

from typing import Dict, Any, List, Optional

from src.health_scoring import compute_health_score
from src.maintenance_recommendations import process_equipment_recommendations
from src.alert_management import create_alert
from src.trend_analysis import analyse_trends, get_dashboard_trend_summary
from src.database import (
    managed_connection,
    save_anomaly_and_risk,
    save_maintenance_recommendation,
    save_alert,
    save_health_score_record,
    get_alerts,
    _rows_to_dicts,
)

# Normal operating ranges (IQR thresholds from normal AI4I 2020 equipment data)
IQR_SENSOR_LIMITS = {
    "Air temperature [K]": (293.50, 306.30),
    "Process temperature [K]": (305.35, 314.55),
    "Rotational speed [rpm]": (1150.0, 1894.0),
    "Torque [Nm]": (13.30, 66.10),
    "Tool wear [min]": (0.0, 322.0),
}


def run_post_prediction_analysis(
    readings: Dict[str, Any],
    prediction_result: Dict[str, Any],
    timestamp: Optional[str] = None,
    save_to_db: bool = True,
) -> Dict[str, Any]:
    """Calculate health score, anomalies, risk level, maintenance actions, and alerts, and persist to DB."""
    equipment_id = str(readings["equipment_id"]).strip()
    equipment_type = str(readings.get("Type", "L")).strip().upper()
    if equipment_type not in ("L", "M", "H"):
        equipment_type = "L"

    air_temp = float(readings["Air temperature [K]"])
    proc_temp = float(readings["Process temperature [K]"])
    rot_speed = float(readings["Rotational speed [rpm]"])
    torque = float(readings["Torque [Nm]"])
    tool_wear = float(readings["Tool wear [min]"])

    failure_probability = float(prediction_result["failure_probability"])

    # 1. Health Score (Member 5 Part C)
    health_score, health_status = compute_health_score(failure_probability)

    # 2. Sensor Anomaly Detection & Risk Classification (Member 2)
    sensor_map = {
        "Air temperature [K]": air_temp,
        "Process temperature [K]": proc_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }
    abnormal_sensors: List[str] = []
    anomalies_list: List[Dict[str, Any]] = []
    for sensor_name, value in sensor_map.items():
        lower, upper = IQR_SENSOR_LIMITS[sensor_name]
        is_abnormal = (value < lower) or (value > upper)
        if is_abnormal:
            abnormal_sensors.append(sensor_name)
            anomalies_list.append(
                {
                    "sensor": sensor_name,
                    "value": value,
                    "normal_min": lower,
                    "normal_max": upper,
                }
            )

    anomaly_count = len(abnormal_sensors)
    abnormal_str = ", ".join(abnormal_sensors) if abnormal_sensors else "None"

    if failure_probability >= 0.70 or anomaly_count >= 2:
        risk_level = "Critical"
        risk_summary = (
            "Critical equipment risk. Detailed equipment inspection recommended. "
            f"Abnormal sensors: {abnormal_str}"
        )
    elif failure_probability >= 0.30 or anomaly_count == 1:
        risk_level = "Warning"
        risk_summary = (
            "Equipment requires monitoring. Check abnormal sensor readings. "
            f"Abnormal sensors: {abnormal_str}"
        )
    else:
        risk_level = "Healthy"
        risk_summary = (
            "Healthy equipment condition. No significant abnormal sensor readings detected."
        )

    # 3. Failure Category Identification & Maintenance Recommendations (Member 3)
    m3_sensor_data = {
        "air_temperature": air_temp,
        "process_temperature": proc_temp,
        "rotational_speed": rot_speed,
        "torque": torque,
        "tool_wear": tool_wear,
        "equipment_type": equipment_type,
    }
    maint_package = process_equipment_recommendations(
        equipment_id=equipment_id,
        sensor_data=m3_sensor_data,
        risk_level=risk_level,
        failure_probability=failure_probability,
    )

    # 4. Automated Alert Generation (Member 4)
    alert_row = {
        "Equipment ID": equipment_id,
        "Risk Level": risk_level,
        "Failure Probability": failure_probability,
        "Abnormal Sensors": abnormal_str,
        "Risk Summary": risk_summary,
    }
    alert_record = create_alert(alert_row, recommendation=maint_package)

    # 5. Persist to Shared Database (Member 5)
    if save_to_db:
        save_health_score_record(
            equipment_id=equipment_id,
            failure_probability=failure_probability,
            health_score=health_score,
            health_status=health_status,
            timestamp=timestamp,
        )
        save_anomaly_and_risk(
            equipment_id=equipment_id,
            anomalies=anomalies_list,
            risk_level=risk_level,
            timestamp=timestamp,
            anomaly_count=anomaly_count,
            abnormal_sensors=abnormal_str,
            risk_summary=risk_summary,
        )
        save_maintenance_recommendation(
            equipment_id=equipment_id,
            primary_issue=maint_package["primary_issue"],
            action_type=maint_package["action_type"],
            recommendations=maint_package["recommended_actions"],
            risk_level=risk_level,
            estimated_downtime=maint_package["estimated_downtime"],
            timestamp=timestamp,
        )
        if alert_record is not None:
            saved_alert_id = save_alert(
                equipment_id=equipment_id,
                risk_level=risk_level,
                failure_probability=failure_probability,
                description=alert_record["alert_description"],
                status="New",
                timestamp=timestamp,
                alert_id=alert_record["alert_id"],
                condition=alert_record["condition"],
            )
            alert_record["alert_id"] = saved_alert_id

    trend_info = analyse_trends(equipment_id)

    return {
        "health_score": health_score,
        "health_status": health_status,
        "risk_level": risk_level,
        "risk_summary": risk_summary,
        "anomaly_count": anomaly_count,
        "abnormal_sensors": abnormal_str,
        "anomalies": anomalies_list,
        "primary_issue": maint_package["primary_issue"],
        "action_type": maint_package["action_type"],
        "recommended_actions": maint_package["recommended_actions"],
        "estimated_downtime": maint_package["estimated_downtime"],
        "alert": alert_record,
        "overall_trend": trend_info["overall_trend"],
        "health_score_trend": trend_info["health_score_trend"],
    }


def get_monitoring_data(page_key: str, limit: int = 100) -> Dict[str, Any]:
    """Fetch stored SQL records for the /monitoring/<page> views."""
    with managed_connection() as (conn, _):
        cur = conn.cursor()

        if page_key == "risk":
            cur.execute(
                """
                SELECT risk_level, COUNT(*) AS cnt
                FROM anomaly_results
                WHERE id IN (SELECT MAX(id) FROM anomaly_results GROUP BY equipment_id)
                GROUP BY risk_level;
                """
            )
            summary = {r["risk_level"]: int(r["cnt"]) for r in cur.fetchall()}

            cur.execute(
                f"""
                SELECT
                    a.equipment_id,
                    a.ts,
                    a.risk_level,
                    a.anomaly_count,
                    a.abnormal_sensors,
                    a.risk_summary,
                    p.failure_probability,
                    p.health_score,
                    p.health_status
                FROM anomaly_results a
                LEFT JOIN (
                    SELECT * FROM predictions
                    WHERE id IN (SELECT MAX(id) FROM predictions GROUP BY equipment_id)
                ) p ON p.equipment_id = a.equipment_id
                WHERE a.id IN (SELECT MAX(id) FROM anomaly_results GROUP BY equipment_id)
                ORDER BY
                    CASE a.risk_level WHEN 'Critical' THEN 1 WHEN 'Warning' THEN 2 ELSE 3 END,
                    p.failure_probability DESC
                LIMIT {int(limit)};
                """
            )
            rows = _rows_to_dicts(cur.fetchall())
            return {"summary": summary, "rows": rows}

        if page_key == "anomalies":
            cur.execute(
                f"""
                SELECT
                    a.equipment_id,
                    a.ts,
                    a.anomaly_count,
                    a.abnormal_sensors,
                    a.risk_level,
                    s.air_temperature,
                    s.process_temperature,
                    s.rotational_speed,
                    s.torque,
                    s.tool_wear
                FROM anomaly_results a
                LEFT JOIN (
                    SELECT * FROM sensor_readings
                    WHERE id IN (SELECT MAX(id) FROM sensor_readings GROUP BY equipment_id)
                ) s ON s.equipment_id = a.equipment_id
                WHERE a.anomaly_count > 0
                ORDER BY a.anomaly_count DESC, a.ts DESC
                LIMIT {int(limit)};
                """
            )
            rows = _rows_to_dicts(cur.fetchall())
            return {"summary": {"Flagged Readings": len(rows)}, "rows": rows}

        if page_key == "alerts":
            alerts = get_alerts(limit=limit)
            summary = {
                "New": sum(1 for a in alerts if a.get("status") == "New"),
                "Acknowledged": sum(1 for a in alerts if a.get("status") == "Acknowledged"),
                "Resolved": sum(1 for a in alerts if a.get("status") == "Resolved"),
            }
            return {"summary": summary, "rows": alerts}

        if page_key == "maintenance":
            cur.execute(
                f"""
                SELECT
                    equipment_id,
                    ts,
                    issue,
                    action_type,
                    recommendation,
                    risk_level,
                    estimated_downtime,
                    status
                FROM maintenance_records
                WHERE issue <> 'Normal'
                ORDER BY
                    CASE risk_level WHEN 'Critical' THEN 1 WHEN 'Warning' THEN 2 ELSE 3 END,
                    ts DESC
                LIMIT {int(limit)};
                """
            )
            rows = _rows_to_dicts(cur.fetchall())
            return {"summary": {"Actionable Recommendations": len(rows)}, "rows": rows}

        if page_key == "history":
            trend_df = get_dashboard_trend_summary(limit=limit)
            rows = trend_df.to_dict("records") if not trend_df.empty else []
            summary = (
                trend_df["overall_trend"].value_counts().to_dict()
                if not trend_df.empty
                else {}
            )
            return {"summary": summary, "rows": rows}

    return {"summary": {}, "rows": []}
"""
Milestone 4 - Member 4: Post-Prediction Analysis & Persistence Service
Aligned with Member 5's exact src/database.py schema and function signatures.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src import database as db
from src.maintenance_recommendations import process_equipment_recommendations

# Import Milestone 3 helpers with safe fallbacks if function names differ
try:
    from src.health_scoring import calculate_health_score
except ImportError:
    def calculate_health_score(equipment_id: str, failure_probability: float) -> Dict[str, Any]:
        score = round(100.0 * (1.0 - failure_probability), 2)
        status = "Good" if score >= 80 else ("Moderate" if score >= 50 else "Poor")
        return {"equipment_id": equipment_id, "health_score": score, "health_status": status}

try:
    from src.anomaly_detection import detect_anomalies_and_risk
except ImportError:
    def detect_anomalies_and_risk(
        equipment_id: str, sensor_data: Dict[str, Any], failure_probability: float
    ) -> Dict[str, Any]:
        # IQR bounds derived from AI4I 2020 dataset
        bounds = {
            "air_temperature": (293.5, 306.3, "Air temperature [K]"),
            "process_temperature": (305.35, 314.55, "Process temperature [K]"),
            "rotational_speed": (1150.0, 1894.0, "Rotational speed [rpm]"),
            "torque": (13.3, 66.1, "Torque [Nm]"),
            "tool_wear": (0.0, 240.0, "Tool wear [min]"),
        }
        abnormal = []
        for key, (low, high, label) in bounds.items():
            val = float(sensor_data.get(key, (low + high) / 2))
            if val < low or val > high:
                abnormal.append(label)

        count = len(abnormal)
        if failure_probability >= 0.70 or count >= 2:
            risk = "Critical"
        elif failure_probability >= 0.30 or count == 1:
            risk = "Warning"
        else:
            risk = "Healthy"

        summary = (
            f"{risk} condition. Abnormal sensors: {', '.join(abnormal)}."
            if abnormal
            else f"{risk} equipment condition."
        )
        return {
            "equipment_id": equipment_id,
            "abnormal_sensors": abnormal,
            "anomaly_count": count,
            "risk_level": risk,
            "risk_summary": summary,
        }


def run_full_post_prediction_analysis(
    equipment_id: str,
    sensor_data: Dict[str, Any],
    predicted_failure: int,
    failure_probability: float,
    timestamp: str = None,
) -> Dict[str, Any]:
    """Runs post-prediction analytics and saves all records to PostgreSQL/SQLite."""
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    db.init_database()

    # 1. Calculate & Save Equipment Health Score
    health_result = calculate_health_score(
        equipment_id=equipment_id,
        failure_probability=failure_probability,
    )
    health_score = float(health_result.get("health_score", round(100.0 * (1.0 - failure_probability), 2)))
    health_status = health_result.get("health_status", "Good")

    db.save_health_score_record(
        equipment_id=equipment_id,
        failure_probability=failure_probability,
        health_score=health_score,
        health_status=health_status,
        timestamp=ts,
    )

    # 2. Detect Sensor Anomalies & Save Risk Classification
    anomaly_result = detect_anomalies_and_risk(
        equipment_id=equipment_id,
        sensor_data=sensor_data,
        failure_probability=failure_probability,
    )
    risk_level = anomaly_result.get("risk_level", "Healthy")
    abnormal_list = anomaly_result.get("abnormal_sensors", [])
    abnormal_str = ", ".join(abnormal_list) if isinstance(abnormal_list, list) else str(abnormal_list)
    anomaly_count = int(anomaly_result.get("anomaly_count", len(abnormal_list) if isinstance(abnormal_list, list) else 0))
    risk_summary = anomaly_result.get("risk_summary", f"{risk_level} risk level")

    db.save_anomaly_and_risk(
        equipment_id=equipment_id,
        anomalies=abnormal_list if isinstance(abnormal_list, list) else abnormal_str,
        risk_level=risk_level,
        timestamp=ts,
        anomaly_count=anomaly_count,
        abnormal_sensors=abnormal_str or "None",
        risk_summary=risk_summary,
    )

    # 3. Generate & Save Maintenance Recommendations
    maint_result = process_equipment_recommendations(
        equipment_id=equipment_id,
        sensor_data=sensor_data,
        risk_level=risk_level,
        failure_probability=failure_probability,
        db_module=None,  # We persist directly below using Member 5's exact signature
    )
    primary_issue = maint_result.get("primary_issue", "Normal")
    action_type = maint_result.get("action_type", "Routine Monitoring")
    rec_actions = maint_result.get("recommended_actions", [])
    est_downtime = maint_result.get("estimated_downtime", "Not specified")

    db.save_maintenance_recommendation(
        equipment_id=equipment_id,
        issue=primary_issue,
        recommendation="; ".join(rec_actions) if isinstance(rec_actions, list) else str(rec_actions),
        risk_level=risk_level,
        timestamp=ts,
        primary_issue=primary_issue,
        action_type=action_type,
        recommendations=rec_actions,
        estimated_downtime=est_downtime,
        status="Pending",
    )

    # 4. Generate & Save Alert (for Warning or Critical risk only)
    alert_id = None
    if risk_level in ("Warning", "Critical"):
        alert_desc = maint_result.get("summary", f"{risk_level} risk detected ({failure_probability * 100:.1f}%).")
        alert_id = db.save_alert(
            equipment_id=equipment_id,
            risk_level=risk_level,
            failure_probability=failure_probability,
            description=alert_desc,
            status="New",
            timestamp=ts,
            condition=primary_issue,
        )

    return {
        "equipment_id": equipment_id,
        "timestamp": ts,
        "health": {
            "health_score": health_score,
            "health_status": health_status,
        },
        "anomaly_and_risk": {
            "risk_level": risk_level,
            "anomaly_count": anomaly_count,
            "abnormal_sensors": abnormal_str or "None",
            "risk_summary": risk_summary,
        },
        "maintenance": maint_result,
        "alert": {
            "alert_id": alert_id,
            "risk_level": risk_level,
            "status": "New" if alert_id else "None",
        },
    }


if __name__ == "__main__":
    sample_telemetry = {
        "air_temperature": 300.5,
        "process_temperature": 310.2,
        "rotational_speed": 1410.0,
        "torque": 67.5,
        "tool_wear": 212.0,
        "equipment_type": "L",
    }
    output = run_full_post_prediction_analysis(
        equipment_id="EQ-101",
        sensor_data=sample_telemetry,
        predicted_failure=1,
        failure_probability=0.875,
    )
    print("\n[✓] Post-Prediction Analysis & DB Save Succeeded:")
    for key, val in output.items():
        print(f"  {key}: {val}")

