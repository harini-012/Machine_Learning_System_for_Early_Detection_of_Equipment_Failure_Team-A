"""
Advanced Task T45 – Propose How the System Could Send Maintenance Alerts
Expected Submission: Alert design and demo
(reports/t45_alert_design_and_demo.md,
 reports/t45_alert_dispatch_log.json,
 reports/t45_alert_dispatch_log.csv)

Implements a multi-channel maintenance alert notification engine supporting:
  1. SMTP HTML Email Work-Order Alerts
  2. Slack / Microsoft Teams Webhook JSON Block-Kit Alerts
  3. PagerDuty / SMS Critical On-Call Escalation Payloads
Includes severity-based routing, deduplication/cooldown suppression, and a
runnable end-to-end dispatch demo.
"""

import json
import os
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Dict, Any, List

import pandas as pd

from src.failure_type_model import REPORTS_DIR


def build_email_alert_payload(alert_event: Dict[str, Any]) -> Dict[str, Any]:
    """Construct an SMTP MIME email payload for maintenance engineering teams."""
    eq_id = alert_event["equipment_id"]
    risk = alert_event["risk_level"]
    prob_pct = round(float(alert_event["failure_probability"]) * 100.0, 2)
    health = alert_event.get("health_score", round(100.0 - prob_pct, 2))
    issue = alert_event.get("primary_issue", "Equipment Failure Risk")
    action = alert_event.get("action_type", "Immediate Inspection")
    downtime = alert_event.get("estimated_downtime", "2-4 hours")
    steps = alert_event.get("recommended_actions", ["Inspect spindle, cooling, and tool wear."])
    shap_summary = alert_event.get("top_shap_driver", "Torque [Nm] & Tool wear [min]")

    subject_prefix = "[CRITICAL SHUTDOWN]" if risk == "Critical" else "[WARNING MAINTENANCE]"
    recipient = (
        os.environ.get("ALERT_CRITICAL_EMAIL", "oncall-reliability@plant-team-a.io")
        if risk == "Critical"
        else os.environ.get("ALERT_WARNING_EMAIL", "shift-maintenance@plant-team-a.io")
    )

    msg = EmailMessage()
    msg["Subject"] = f"{subject_prefix} {eq_id} - {issue} (Failure Prob: {prob_pct:.1f}%)"
    msg["From"] = os.environ.get("ALERT_SENDER_EMAIL", "predictive-maintenance-bot@plant-team-a.io")
    msg["To"] = recipient

    steps_bullets = "\n".join(f"  - {s}" for s in steps)
    body_text = (
        f"PREDICTIVE MAINTENANCE AUTOMATED WORK ORDER\n"
        f"===========================================\n"
        f"Equipment ID       : {eq_id}\n"
        f"Severity / Risk    : {risk}\n"
        f"Failure Probability: {prob_pct:.2f}%\n"
        f"Health Score       : {health:.2f} / 100\n"
        f"Suspected Failure  : {issue}\n"
        f"Top SHAP Driver    : {shap_summary}\n"
        f"Required Action    : {action} (Est. Downtime: {downtime})\n\n"
        f"Recommended Maintenance Steps:\n{steps_bullets}\n\n"
        f"Dashboard Link     : https://machine-learning-system-for-early.onrender.com/equipment/{eq_id}\n"
    )
    msg.set_content(body_text)

    return {
        "channel": "SMTP_EMAIL",
        "recipient": recipient,
        "subject": msg["Subject"],
        "sla": "15 minutes (Immediate Dispatch)" if risk == "Critical" else "24 hours (Next Shift Window)",
        "body_preview": body_text,
    }


def build_slack_webhook_payload(alert_event: Dict[str, Any]) -> Dict[str, Any]:
    """Construct a Slack Block-Kit webhook payload for real-time plant floor channels."""
    eq_id = alert_event["equipment_id"]
    risk = alert_event["risk_level"]
    prob_pct = round(float(alert_event["failure_probability"]) * 100.0, 2)
    health = alert_event.get("health_score", round(100.0 - prob_pct, 2))
    issue = alert_event.get("primary_issue", "Equipment Failure Risk")
    action = alert_event.get("action_type", "Immediate Inspection")
    abnormal = alert_event.get("abnormal_sensors", "None")

    channel = "#critical-plant-alerts" if risk == "Critical" else "#maintenance-warnings"
    color = "#e11d48" if risk == "Critical" else "#d97706"

    payload = {
        "channel": channel,
        "username": "AI4I Predictive Maintenance Bot",
        "attachments": [
            {
                "color": color,
                "title": f"{risk.upper()} ALERT: Equipment {eq_id} ({issue})",
                "title_link": f"https://machine-learning-system-for-early.onrender.com/equipment/{eq_id}",
                "fields": [
                    {"title": "Failure Probability", "value": f"{prob_pct:.2f}%", "short": True},
                    {"title": "Health Score", "value": f"{health:.2f} / 100", "short": True},
                    {"title": "Abnormal Sensors", "value": abnormal, "short": True},
                    {"title": "Prescribed Action", "value": action, "short": True},
                ],
                "footer": "XGBoost + TreeSHAP Predictive Maintenance Engine",
                "ts": int(datetime.now(timezone.utc).timestamp()),
            }
        ],
    }
    return {
        "channel": "SLACK_WEBHOOK",
        "target_channel": channel,
        "payload": payload,
    }


def build_pagerduty_sms_payload(alert_event: Dict[str, Any]) -> Dict[str, Any]:
    """Construct a PagerDuty Events v2 / SMS escalation payload for Critical alerts."""
    eq_id = alert_event["equipment_id"]
    risk = alert_event["risk_level"]
    prob_pct = round(float(alert_event["failure_probability"]) * 100.0, 2)
    issue = alert_event.get("primary_issue", "Equipment Failure Risk")

    return {
        "channel": "PAGERDUTY_SMS",
        "routing_key": "PD-SERVICE-CNC-SPINDLE-TEAM-A",
        "event_action": "trigger",
        "dedup_key": f"pm-alert-{eq_id}-{issue.lower().replace(' ', '-')}",
        "SMS_text": (
            f"[PLANT-ALERT {risk.upper()}] {eq_id}: {issue} detected "
            f"(Prob {prob_pct:.1f}%). Inspect immediately: "
            f"https://machine-learning-system-for-early.onrender.com/equipment/{eq_id}"
        ),
    }


def dispatch_maintenance_alert(
    alert_event: Dict[str, Any],
    active_open_alerts: Dict[str, str] = None,
) -> Dict[str, Any]:
    """Route and dispatch a maintenance alert across applicable channels with deduplication."""
    active_open_alerts = active_open_alerts if active_open_alerts is not None else {}
    eq_id = alert_event["equipment_id"]
    risk = alert_event["risk_level"]
    ts = alert_event.get("timestamp") or datetime.now(timezone.utc).isoformat(timespec="seconds")

    if risk not in ("Warning", "Critical"):
        return {
            "equipment_id": eq_id,
            "timestamp": ts,
            "risk_level": risk,
            "dispatched": False,
            "reason": "Healthy equipment; no alert notification required.",
            "channels_triggered": [],
        }

    # Deduplication & Escalation Gate
    prev_risk = active_open_alerts.get(eq_id)
    if prev_risk == risk or (prev_risk == "Critical" and risk == "Warning"):
        return {
            "equipment_id": eq_id,
            "timestamp": ts,
            "risk_level": risk,
            "dispatched": False,
            "reason": f"Suppressed duplicate alert (open {prev_risk} alert already active for {eq_id}).",
            "channels_triggered": [],
        }

    active_open_alerts[eq_id] = risk

    email_info = build_email_alert_payload(alert_event)
    slack_info = build_slack_webhook_payload(alert_event)
    channels = ["SMTP_EMAIL", "SLACK_WEBHOOK"]
    payloads = {"email": email_info, "slack": slack_info}

    if risk == "Critical":
        pd_info = build_pagerduty_sms_payload(alert_event)
        channels.append("PAGERDUTY_SMS")
        payloads["pagerduty_sms"] = pd_info

    return {
        "equipment_id": eq_id,
        "timestamp": ts,
        "risk_level": risk,
        "failure_probability": round(float(alert_event["failure_probability"]), 4),
        "health_score": alert_event.get("health_score", round(100.0 * (1.0 - float(alert_event["failure_probability"])), 2)),
        "primary_issue": alert_event.get("primary_issue", "Equipment Failure Risk"),
        "action_type": alert_event.get("action_type", "Immediate Inspection"),
        "dispatched": True,
        "reason": "Escalated Warning -> Critical" if prev_risk == "Warning" else "New Active Alert Dispatched",
        "channels_triggered": channels,
        "sla": email_info["sla"],
        "recipient": email_info["recipient"],
        "payloads": payloads,
    }


def run_alert_notification_demo() -> Dict[str, Any]:
    """Run a multi-event alert notification demo showcasing routing, escalation, and deduplication."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    demo_events = [
        {
            "equipment_id": "EQ-CNC-104",
            "timestamp": "2026-11-19T14:00:00+00:00",
            "risk_level": "Warning",
            "failure_probability": 0.4215,
            "health_score": 57.85,
            "primary_issue": "Tool Wear Failure (TWF)",
            "action_type": "Preventive Tool Replacement",
            "estimated_downtime": "45 minutes",
            "abnormal_sensors": "Tool wear [min]",
            "top_shap_driver": "Tool wear [min] = 202.0 (+1.48 SHAP)",
            "recommended_actions": [
                "Schedule cutting tool replacement at end of current batch.",
                "Verify spindle runout and lubrication flow.",
            ],
        },
        {
            "equipment_id": "EQ-CNC-104",
            "timestamp": "2026-11-19T14:15:00+00:00",
            "risk_level": "Warning",
            "failure_probability": 0.4490,
            "health_score": 55.10,
            "primary_issue": "Tool Wear Failure (TWF)",
            "action_type": "Preventive Tool Replacement",
            "estimated_downtime": "45 minutes",
            "abnormal_sensors": "Tool wear [min]",
            "top_shap_driver": "Tool wear [min] = 206.0 (+1.56 SHAP)",
            "recommended_actions": ["Schedule cutting tool replacement."],
        },
        {
            "equipment_id": "EQ-CNC-104",
            "timestamp": "2026-11-19T14:30:00+00:00",
            "risk_level": "Critical",
            "failure_probability": 0.9180,
            "health_score": 8.20,
            "primary_issue": "Overstrain Failure (OSF)",
            "action_type": "Immediate Shutdown & Overhaul",
            "estimated_downtime": "3-4 hours",
            "abnormal_sensors": "Torque [Nm], Tool wear [min]",
            "top_shap_driver": "Torque [Nm] = 66.8 (+2.64 SHAP), Tool wear [min] = 224.0 (+2.11 SHAP)",
            "recommended_actions": [
                "Halt spindle immediately to prevent shaft fracture.",
                "Replace worn cutting tool and recalibrate torque limiter.",
            ],
        },
        {
            "equipment_id": "EQ-MILL-209",
            "timestamp": "2026-11-19T14:45:00+00:00",
            "risk_level": "Critical",
            "failure_probability": 0.8740,
            "health_score": 12.60,
            "primary_issue": "Heat Dissipation Failure (HDF)",
            "action_type": "Emergency Cooling System Service",
            "estimated_downtime": "2-3 hours",
            "abnormal_sensors": "Process temperature [K], Rotational speed [rpm]",
            "top_shap_driver": "Temperature Difference [K] = 7.8 (+2.31 SHAP)",
            "recommended_actions": [
                "Inspect coolant pump pressure and clear heat exchanger fins.",
                "Verify ambient ventilation before restarting spindle.",
            ],
        },
    ]

    open_alerts: Dict[str, str] = {}
    dispatch_log: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    for ev in demo_events:
        res = dispatch_maintenance_alert(ev, active_open_alerts=open_alerts)
        dispatch_log.append(res)
        csv_rows.append(
            {
                "Timestamp": res["timestamp"],
                "Equipment ID": res["equipment_id"],
                "Risk Level": res["risk_level"],
                "Failure Probability": f"{ev['failure_probability']*100:.2f}%",
                "Suspected Issue": ev["primary_issue"],
                "Dispatched": "Yes" if res["dispatched"] else "Suppressed (Dedup)",
                "Channels Triggered": ", ".join(res["channels_triggered"]) if res["channels_triggered"] else "None",
                "Target / SLA": f"{res.get('recipient', 'N/A')} ({res.get('sla', 'Suppressed')})",
                "Dispatch Reason": res["reason"],
            }
        )

    json_path = REPORTS_DIR / "t45_alert_dispatch_log.json"
    json_path.write_text(json.dumps(dispatch_log, indent=2), encoding="utf-8")

    csv_path = REPORTS_DIR / "t45_alert_dispatch_log.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    sample_email = dispatch_log[2]["payloads"]["email"]["body_preview"]
    sample_slack = json.dumps(dispatch_log[2]["payloads"]["slack"]["payload"], indent=2)
    sample_sms = dispatch_log[2]["payloads"]["pagerduty_sms"]["SMS_text"]

    md_table = "\n".join(
        f"| `{r['Timestamp']}` | **{r['Equipment ID']}** | `{r['Risk Level']}` | `{r['Failure Probability']}` | {r['Suspected Issue']} | **{r['Dispatched']}** | `{r['Channels Triggered']}` | {r['Dispatch Reason']} |"
        for r in csv_rows
    )

    md_content = f"""# Advanced Task T45: Multi-Channel Maintenance Alert Design & Live Demo

## 1. Alert Notification Architecture & Routing Policy

To ensure maintenance teams receive timely, actionable notifications without suffering from alert fatigue, the system implements a **Three-Tier Multi-Channel Alert Dispatcher** (`src/maintenance_alert_notifier.py`):

| Risk Tier | Trigger Condition | Notification Channels | Target Audience & SLA | Deduplication & Escalation Rule |
| :--- | :--- | :--- | :--- | :--- |
| **Healthy** | $P < 0.30$ and $0$ sensor anomalies | None (Logged to SQL DB) | Dashboard monitoring only | Resets cooldown timer |
| **Warning** | $0.30 \\le P < 0.70$ or $1$ abnormal sensor | • **SMTP Email Work Order**<br/>• **Slack `#maintenance-warnings`** | Shift Maintenance Queue (`SLA: 24 hours`) | Suppresses repeat `Warning` emails while an alert is `New` or `Acknowledged` |
| **Critical** | $P \\ge 0.70$ or $\\ge 2$ abnormal sensors | • **Urgent SMTP Email**<br/>• **Slack `#critical-plant-alerts`**<br/>• **PagerDuty / SMS On-Call Page** | On-Call Reliability Engineer (`SLA: <15 minutes`) | Immediately overrides/escalates any open `Warning` alert |

---

## 2. Live Dispatch & Deduplication Demo Log

| Timestamp | Equipment ID | Risk Level | Failure Prob | Suspected Issue | Dispatched? | Channels Triggered | Dispatch Reason |
| :---: | :---: | :---: | :---: | :--- | :---: | :--- | :--- |
{md_table}

---

## 3. Sample Dispatched Payloads (Critical Escalation Event for `EQ-CNC-104`)

### A. Automated SMTP Email Work Order (`SMTP_EMAIL`)
```text
{sample_email}```

### B. Slack Block-Kit Webhook Payload (`SLACK_WEBHOOK`)
```json
{sample_slack}
```

### C. PagerDuty / SMS On-Call Escalation (`PAGERDUTY_SMS`)
```text
{sample_sms}
```
"""
    md_path = REPORTS_DIR / "t45_alert_design_and_demo.md"
    md_path.write_text(md_content, encoding="utf-8")

    return {
        "json_path": str(json_path),
        "csv_path": str(csv_path),
        "md_path": str(md_path),
        "csv_rows": csv_rows,
        "dispatch_log": dispatch_log,
    }


if __name__ == "__main__":
    res = run_alert_notification_demo()
    print("T45 Alert Design & Demo saved to:", res["md_path"])
