"""CSV downloads and alert status updates for the Flask dashboard."""

import csv
import io
from pathlib import Path

from flask import Blueprint, Response, flash, redirect, request, send_file, url_for

from src import database as db


REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
report_bp = Blueprint("reports", __name__)


def _csv_response(headers, rows, filename):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(rows)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@report_bp.get("/reports/download/predicted-failures")
def download_predicted_failures():
    headers = [
        "Equipment ID",
        "Timestamp",
        "Predicted Failure",
        "Prediction Label",
        "Failure Probability",
        "Health Score",
        "Health Status",
    ]

    with db.managed_connection() as (conn, _backend):
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT
                    equipment_id,
                    ts,
                    failure_predicted,
                    prediction_label,
                    failure_probability,
                    health_score,
                    health_status
                FROM predictions
                WHERE failure_predicted = TRUE
                ORDER BY ts DESC, id DESC
                """
            )
            records = db._rows_to_dicts(cursor.fetchall())
        finally:
            cursor.close()

    rows = [
        (
            record["equipment_id"],
            record["ts"],
            record["failure_predicted"],
            record["prediction_label"],
            record["failure_probability"],
            record["health_score"],
            record["health_status"],
        )
        for record in records
    ]
    return _csv_response(headers, rows, "predicted_failures_report.csv")


@report_bp.get("/reports/download/maintenance-activities")
def download_maintenance_activities():
    headers = [
        "Equipment ID",
        "Timestamp",
        "Risk Level",
        "Issue",
        "Action Type",
        "Recommendation",
        "Estimated Downtime",
        "Status",
        "Completed At",
    ]

    with db.managed_connection() as (conn, _backend):
        cursor = conn.cursor()
        try:
            cursor.execute(
                """
                SELECT
                    equipment_id,
                    ts,
                    risk_level,
                    issue,
                    action_type,
                    recommendation,
                    estimated_downtime,
                    status,
                    completed_at
                FROM maintenance_records
                ORDER BY ts DESC, id DESC
                """
            )
            records = db._rows_to_dicts(cursor.fetchall())
        finally:
            cursor.close()

    rows = [
        (
            record["equipment_id"],
            record["ts"],
            record["risk_level"],
            record["issue"],
            record["action_type"],
            record["recommendation"],
            record["estimated_downtime"],
            record["status"],
            record["completed_at"],
        )
        for record in records
    ]
    return _csv_response(headers, rows, "maintenance_activities_report.csv")


@report_bp.get("/reports/download/model-performance")
def download_model_performance():
    for filename in ("heldout_test_verification.csv", "model_comparison.csv"):
        csv_path = REPORTS_DIR / filename
        if csv_path.is_file():
            return send_file(
                csv_path,
                mimetype="text/csv",
                as_attachment=True,
                download_name="model_performance_report.csv",
            )

    return Response(
        "Model performance report is unavailable. Run the held-out evaluation first.",
        status=404,
        mimetype="text/plain",
    )


@report_bp.post("/alerts/<alert_id>/status")
def update_alert_status(alert_id):
    new_status = request.form.get("status", "").strip()

    if new_status not in {"New", "Acknowledged", "Resolved"}:
        flash("Select a valid alert status.", "error")
        return redirect(url_for("monitoring", page="alerts"), code=303)

    try:
        db.update_alert_status_in_db(
            alert_id=alert_id,
            new_status=new_status,
        )
    except ValueError as exc:
        flash(str(exc), "error")
    except Exception:
        report_bp.logger.exception("Alert status update failed")
        flash("Unable to update the alert status.", "error")
    else:
        flash(f"Alert status updated to {new_status}.", "success")

    return redirect(url_for("monitoring", page="alerts"), code=303)