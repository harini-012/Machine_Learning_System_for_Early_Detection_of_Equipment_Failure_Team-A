"""
Member 5 – Part D: Historical Trend Analysis (src/trend_analysis.py)

Retrieves historical equipment readings, predictions, health scores, and alerts
from the shared database and analyses changes in:
    1. Sensor readings (Air temp, Process temp, Rotational speed, Torque, Tool wear)
    2. Failure probability
    3. Equipment health score
    4. Number of alerts

Classifies equipment trends into:
    - Improving
    - Stable
    - Deteriorating

Provides historical series and aggregated analytics for the Milestone 4 Web Dashboard.
"""

import logging
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

try:
    from .database import retrieve_equipment_history, managed_connection
except ImportError:
    from database import retrieve_equipment_history, managed_connection

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
TREND_OUTPUT_FILE = PROJECT_DIR / "reports" / "trend_analysis_summary.csv"


def _parse_ts(value: Any) -> Optional[datetime.datetime]:
    """Parse an ISO timestamp string or datetime into a UTC-aware datetime."""
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=datetime.timezone.utc)
        return value.astimezone(datetime.timezone.utc)
    try:
        dt = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc)
    except Exception:
        return None


def _trend_lower_is_better(delta: float, tolerance: float = 0.0) -> str:
    """Classify trend for metrics where a lower value is healthier (e.g. failure probability, tool wear).

    - delta < -tolerance -> Improving
    - delta > +tolerance -> Deteriorating
    - otherwise          -> Stable
    """
    if delta < -abs(tolerance):
        return "Improving"
    if delta > abs(tolerance):
        return "Deteriorating"
    return "Stable"


def _trend_higher_is_better(delta: float, tolerance: float = 0.0) -> str:
    """Classify trend for metrics where a higher value is healthier (e.g. health score).

    - delta > +tolerance -> Improving
    - delta < -tolerance -> Deteriorating
    - otherwise          -> Stable
    """
    if delta > abs(tolerance):
        return "Improving"
    if delta < -abs(tolerance):
        return "Deteriorating"
    return "Stable"


def analyse_trends(equipment_id: str, lookback_days: int = 30) -> Dict[str, Any]:
    """Analyse historical trends for a specific piece of equipment.

    Retrieves sensor readings, predictions, health scores, and alerts from the
    shared database over the past `lookback_days` and classifies each metric
    and the overall equipment trajectory as 'Improving', 'Stable', or 'Deteriorating'.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    start = now - datetime.timedelta(days=lookback_days)

    history = retrieve_equipment_history(equipment_id)

    def _in_window(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered = []
        for r in rows:
            ts = _parse_ts(r.get("ts"))
            if ts is None or ts >= start:
                filtered.append(r)
        return filtered

    # 1. Sensor readings analysis
    sensor_rows = _in_window(history.get("sensor_readings", []))
    sensor_metrics: Dict[str, Dict[str, Any]] = {}
    sensor_tolerances = {
        "air_temperature": 0.5,
        "process_temperature": 0.5,
        "rotational_speed": 25.0,
        "torque": 2.0,
        "tool_wear": 5.0,
    }

    for col, tol in sensor_tolerances.items():
        vals = [float(r[col]) for r in sensor_rows if col in r and r[col] is not None]
        if len(vals) >= 2:
            delta = vals[-1] - vals[0]
            trend_label = _trend_lower_is_better(delta, tolerance=tol)
        elif len(vals) == 1:
            delta = 0.0
            trend_label = "Stable"
        else:
            delta = 0.0
            trend_label = "Stable"

        sensor_metrics[col] = {
            "latest": round(vals[-1], 2) if vals else None,
            "earliest": round(vals[0], 2) if vals else None,
            "mean": round(sum(vals) / len(vals), 2) if vals else None,
            "delta": round(delta, 2),
            "trend": trend_label,
        }

    if not sensor_rows:
        sensor_trend = "Stable"
    else:
        # Overall sensor trend driven primarily by tool wear, torque, and air temp
        sub_trends = [
            sensor_metrics["tool_wear"]["trend"],
            sensor_metrics["torque"]["trend"],
            sensor_metrics["air_temperature"]["trend"],
        ]
        if "Deteriorating" in sub_trends:
            sensor_trend = "Deteriorating"
        elif "Improving" in sub_trends:
            sensor_trend = "Improving"
        else:
            sensor_trend = "Stable"

    # 2. Failure probability trend
    pred_rows = _in_window(history.get("predictions", []))
    probs = [
        float(r["failure_probability"])
        for r in pred_rows
        if "failure_probability" in r and r["failure_probability"] is not None
    ]
    if len(probs) >= 2:
        prob_delta = probs[-1] - probs[0]
        prob_trend = _trend_lower_is_better(prob_delta, tolerance=0.05)
    elif len(probs) == 1:
        prob_delta = 0.0
        prob_trend = "Deteriorating" if probs[-1] >= 0.50 else "Stable"
    else:
        prob_delta = 0.0
        prob_trend = "Stable"

    # 3. Equipment health score trend (higher is better!)
    score_rows = _in_window(history.get("health_scores", []))
    if score_rows:
        health_scores = [
            float(r["health_score"])
            for r in score_rows
            if "health_score" in r and r["health_score"] is not None
        ]
    else:
        health_scores = [100.0 * (1.0 - p) for p in probs]

    if len(health_scores) >= 2:
        health_delta = health_scores[-1] - health_scores[0]
        health_trend = _trend_higher_is_better(health_delta, tolerance=5.0)
    elif len(health_scores) == 1:
        health_delta = 0.0
        health_trend = "Deteriorating" if health_scores[-1] < 50.0 else "Stable"
    else:
        health_delta = 0.0
        health_trend = "Stable"

    # 4. Alerts trend
    alert_rows = _in_window(history.get("alerts", []))
    alert_count = len(alert_rows)
    active_alerts = [r for r in alert_rows if r.get("status") in ("New", "Acknowledged")]
    resolved_alerts = [r for r in alert_rows if r.get("status") == "Resolved"]

    if len(active_alerts) > 0:
        alerts_trend = "Deteriorating"
    elif len(resolved_alerts) > 0 and len(active_alerts) == 0:
        alerts_trend = "Improving"
    else:
        alerts_trend = "Stable"

    # 5. Overall Equipment Trend classification (Improving / Stable / Deteriorating)
    indicators = [prob_trend, health_trend, sensor_trend, alerts_trend]
    if indicators.count("Deteriorating") >= 1 and prob_trend == "Deteriorating":
        overall_trend = "Deteriorating"
    elif indicators.count("Deteriorating") >= 2:
        overall_trend = "Deteriorating"
    elif indicators.count("Improving") >= 1 and "Deteriorating" not in indicators:
        overall_trend = "Improving"
    else:
        overall_trend = "Stable"

    return {
        "equipment_id": equipment_id,
        "lookback_days": lookback_days,
        "overall_trend": overall_trend,
        "sensor_trend": sensor_trend,
        "failure_probability_trend": prob_trend,
        "health_score_trend": health_trend,
        "alerts_trend": alerts_trend,
        "latest_failure_probability": round(probs[-1], 4) if probs else None,
        "latest_health_score": round(health_scores[-1], 2) if health_scores else None,
        "failure_probability_delta": round(prob_delta, 4),
        "health_score_delta": round(health_delta, 2),
        "alert_count": alert_count,
        "active_alert_count": len(active_alerts),
        "sensor_metrics": sensor_metrics,
        "history": {
            "sensor_readings": sensor_rows,
            "predictions": pred_rows,
            "health_scores": score_rows,
            "alerts": alert_rows,
        },
    }


def get_dashboard_trend_summary(limit: int = 500) -> pd.DataFrame:
    """Generate a fleet-wide trend summary table for the Milestone 4 Web Dashboard."""
    with managed_connection() as (conn, _):
        cur = conn.cursor()
        cur.execute(
            f"SELECT equipment_id FROM equipment ORDER BY equipment_id ASC LIMIT {int(limit)};"
        )
        rows = cur.fetchall()
        equipment_ids = [
            r["equipment_id"] if isinstance(r, dict) or hasattr(r, "keys") else r[0]
            for r in rows
        ]

    records = []
    for eq_id in equipment_ids:
        res = analyse_trends(eq_id)
        records.append(
            {
                "equipment_id": res["equipment_id"],
                "overall_trend": res["overall_trend"],
                "sensor_trend": res["sensor_trend"],
                "failure_probability_trend": res["failure_probability_trend"],
                "health_score_trend": res["health_score_trend"],
                "alerts_trend": res["alerts_trend"],
                "latest_failure_probability": res["latest_failure_probability"],
                "latest_health_score": res["latest_health_score"],
                "alert_count": res["alert_count"],
                "active_alert_count": res["active_alert_count"],
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    print("\n==================================================")
    print("MEMBER 5 (PART D): HISTORICAL TREND ANALYSIS")
    print("==================================================")

    summary_df = get_dashboard_trend_summary(limit=2000)
    if summary_df.empty:
        print("No equipment records found in database yet. Run the pipeline first.")
        return

    TREND_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(TREND_OUTPUT_FILE, index=False)

    print(f"\nTotal equipment analysed: {len(summary_df)}")
    print("\nOverall Equipment Trend Distribution:")
    print(summary_df["overall_trend"].value_counts().to_string())

    print("\nSample Trend Analysis Results (first 10 records):")
    print(summary_df.head(10).to_string(index=False))
    print(f"\nTrend analysis summary saved to: {TREND_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
