# Advanced Task T45: Multi-Channel Maintenance Alert Design & Live Demo

## 1. Alert Notification Architecture & Routing Policy

To ensure maintenance teams receive timely, actionable notifications without suffering from alert fatigue, the system implements a **Three-Tier Multi-Channel Alert Dispatcher** (`src/maintenance_alert_notifier.py`):

| Risk Tier | Trigger Condition | Notification Channels | Target Audience & SLA | Deduplication & Escalation Rule |
| :--- | :--- | :--- | :--- | :--- |
| **Healthy** | $P < 0.30$ and $0$ sensor anomalies | None (Logged to SQL DB) | Dashboard monitoring only | Resets cooldown timer |
| **Warning** | $0.30 \le P < 0.70$ or $1$ abnormal sensor | • **SMTP Email Work Order**<br/>• **Slack `#maintenance-warnings`** | Shift Maintenance Queue (`SLA: 24 hours`) | Suppresses repeat `Warning` emails while an alert is `New` or `Acknowledged` |
| **Critical** | $P \ge 0.70$ or $\ge 2$ abnormal sensors | • **Urgent SMTP Email**<br/>• **Slack `#critical-plant-alerts`**<br/>• **PagerDuty / SMS On-Call Page** | On-Call Reliability Engineer (`SLA: <15 minutes`) | Immediately overrides/escalates any open `Warning` alert |

---

## 2. Live Dispatch & Deduplication Demo Log

| Timestamp | Equipment ID | Risk Level | Failure Prob | Suspected Issue | Dispatched? | Channels Triggered | Dispatch Reason |
| :---: | :---: | :---: | :---: | :--- | :---: | :--- | :--- |
| `2026-11-19T14:00:00+00:00` | **EQ-CNC-104** | `Warning` | `42.15%` | Tool Wear Failure (TWF) | **Yes** | `SMTP_EMAIL, SLACK_WEBHOOK` | New Active Alert Dispatched |
| `2026-11-19T14:15:00+00:00` | **EQ-CNC-104** | `Warning` | `44.90%` | Tool Wear Failure (TWF) | **Suppressed (Dedup)** | `None` | Suppressed duplicate alert (open Warning alert already active for EQ-CNC-104). |
| `2026-11-19T14:30:00+00:00` | **EQ-CNC-104** | `Critical` | `91.80%` | Overstrain Failure (OSF) | **Yes** | `SMTP_EMAIL, SLACK_WEBHOOK, PAGERDUTY_SMS` | Escalated Warning -> Critical |
| `2026-11-19T14:45:00+00:00` | **EQ-MILL-209** | `Critical` | `87.40%` | Heat Dissipation Failure (HDF) | **Yes** | `SMTP_EMAIL, SLACK_WEBHOOK, PAGERDUTY_SMS` | New Active Alert Dispatched |

---

## 3. Sample Dispatched Payloads (Critical Escalation Event for `EQ-CNC-104`)

### A. Automated SMTP Email Work Order (`SMTP_EMAIL`)
```text
PREDICTIVE MAINTENANCE AUTOMATED WORK ORDER
===========================================
Equipment ID       : EQ-CNC-104
Severity / Risk    : Critical
Failure Probability: 91.80%
Health Score       : 8.20 / 100
Suspected Failure  : Overstrain Failure (OSF)
Top SHAP Driver    : Torque [Nm] = 66.8 (+2.64 SHAP), Tool wear [min] = 224.0 (+2.11 SHAP)
Required Action    : Immediate Shutdown & Overhaul (Est. Downtime: 3-4 hours)

Recommended Maintenance Steps:
  - Halt spindle immediately to prevent shaft fracture.
  - Replace worn cutting tool and recalibrate torque limiter.

Dashboard Link     : https://machine-learning-system-for-early.onrender.com/equipment/EQ-CNC-104
```

### B. Slack Block-Kit Webhook Payload (`SLACK_WEBHOOK`)
```json
{
  "channel": "#critical-plant-alerts",
  "username": "AI4I Predictive Maintenance Bot",
  "attachments": [
    {
      "color": "#e11d48",
      "title": "CRITICAL ALERT: Equipment EQ-CNC-104 (Overstrain Failure (OSF))",
      "title_link": "https://machine-learning-system-for-early.onrender.com/equipment/EQ-CNC-104",
      "fields": [
        {
          "title": "Failure Probability",
          "value": "91.80%",
          "short": true
        },
        {
          "title": "Health Score",
          "value": "8.20 / 100",
          "short": true
        },
        {
          "title": "Abnormal Sensors",
          "value": "Torque [Nm], Tool wear [min]",
          "short": true
        },
        {
          "title": "Prescribed Action",
          "value": "Immediate Shutdown & Overhaul",
          "short": true
        }
      ],
      "footer": "XGBoost + TreeSHAP Predictive Maintenance Engine",
      "ts": 1795102019
    }
  ]
}
```

### C. PagerDuty / SMS On-Call Escalation (`PAGERDUTY_SMS`)
```text
[PLANT-ALERT CRITICAL] EQ-CNC-104: Overstrain Failure (OSF) detected (Prob 91.8%). Inspect immediately: https://machine-learning-system-for-early.onrender.com/equipment/EQ-CNC-104
```
