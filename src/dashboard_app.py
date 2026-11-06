"""
Predictive Maintenance, Health Scoring & Trend Analysis Web Dashboard
(Milestone 3 Member 5 + Milestone 4 Preview)

Connects directly to the shared database (src/database.py), health scoring
(src/health_scoring.py), trend analysis (src/trend_analysis.py), and ML
pipeline (Members 1-4) to serve a live interactive web dashboard at:
    http://localhost:8050

Also exports:
    - reports/milestone3_dashboard.html (standalone interactive HTML dashboard)
    - reports/member5_dashboard_overview.png (high-resolution summary chart)
"""

import json
import urllib.parse
from pathlib import Path
from wsgiref.simple_server import make_server
import os
from typing import Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    from .database import (
        managed_connection,
        get_alerts,
        update_alert_status_in_db,
        retrieve_equipment_history,
        get_table_counts,
        get_database_backend,
    )
    from .health_scoring import compute_health_score, evaluate_equipment_health
    from .trend_analysis import analyse_trends, get_dashboard_trend_summary
    from .model_integration import load_best_model, predict_failure
    from .maintenance_recommendations import process_equipment_recommendations
except ImportError:
    from database import (
        managed_connection,
        get_alerts,
        update_alert_status_in_db,
        retrieve_equipment_history,
        get_table_counts,
        get_database_backend,
    )
    from health_scoring import compute_health_score, evaluate_equipment_health
    from trend_analysis import analyse_trends, get_dashboard_trend_summary
    from model_integration import load_best_model, predict_failure
    from maintenance_recommendations import process_equipment_recommendations

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_DIR / "reports"
DASHBOARD_HTML_PATH = REPORTS_DIR / "milestone3_dashboard.html"
DASHBOARD_PNG_PATH = REPORTS_DIR / "member5_dashboard_overview.png"


def fetch_dashboard_payload() -> Dict[str, Any]:
    """Query the SQL database and assemble the complete dashboard dataset."""
    counts = get_table_counts()
    backend = get_database_backend()

    with managed_connection() as (conn, _):
        cur = conn.cursor()

        # Health status breakdown (latest per equipment)
        cur.execute(
            """
            SELECT health_status, COUNT(*) AS cnt
            FROM predictions
            WHERE id IN (SELECT MAX(id) FROM predictions GROUP BY equipment_id)
            GROUP BY health_status;
            """
        )
        health_dist = {row["health_status"]: int(row["cnt"]) for row in cur.fetchall()}

        # Risk level breakdown
        cur.execute(
            """
            SELECT risk_level, COUNT(*) AS cnt
            FROM anomaly_results
            GROUP BY risk_level;
            """
        )
        risk_dist = {row["risk_level"]: int(row["cnt"]) for row in cur.fetchall()}

        # Maintenance issue breakdown
        cur.execute(
            """
            SELECT issue, COUNT(*) AS cnt
            FROM maintenance_records
            WHERE issue <> 'Normal'
            GROUP BY issue
            ORDER BY cnt DESC;
            """
        )
        issue_dist = {row["issue"]: int(row["cnt"]) for row in cur.fetchall()}

        # Average health score & failure probability
        cur.execute(
            """
            SELECT AVG(health_score) AS avg_health, AVG(failure_probability) AS avg_prob
            FROM predictions
            WHERE id IN (SELECT MAX(id) FROM predictions GROUP BY equipment_id);
            """
        )
        avg_row = cur.fetchone()
        avg_health = round(float(avg_row["avg_health"] or 0.0), 2)
        avg_prob = round(float(avg_row["avg_prob"] or 0.0), 4)

        # Joined equipment status table (top critical/warning first, plus sample healthy)
        cur.execute(
            """
            SELECT
                a.equipment_id,
                e.product_id,
                e.equipment_type,
                s.air_temperature,
                s.process_temperature,
                s.rotational_speed,
                s.torque,
                s.tool_wear,
                p.failure_probability,
                p.health_score,
                p.health_status,
                a.anomaly_count,
                a.abnormal_sensors,
                a.risk_level,
                m.issue AS primary_issue,
                m.action_type,
                m.recommendation,
                m.estimated_downtime
            FROM anomaly_results a
            JOIN equipment e ON e.equipment_id = a.equipment_id
            LEFT JOIN (
                SELECT * FROM sensor_readings
                WHERE id IN (SELECT MAX(id) FROM sensor_readings GROUP BY equipment_id)
            ) s ON s.equipment_id = a.equipment_id
            LEFT JOIN (
                SELECT * FROM predictions
                WHERE id IN (SELECT MAX(id) FROM predictions GROUP BY equipment_id)
            ) p ON p.equipment_id = a.equipment_id
            LEFT JOIN maintenance_records m ON m.equipment_id = a.equipment_id
            ORDER BY p.failure_probability DESC, a.anomaly_count DESC
            LIMIT 250;
            """
        )
        equipment_rows = [dict(r) for r in cur.fetchall()]

    # Trend summary from trend_analysis_summary.csv or live query
    trend_csv = REPORTS_DIR / "trend_analysis_summary.csv"
    if trend_csv.exists():
        t_df = pd.read_csv(trend_csv)
    else:
        t_df = get_dashboard_trend_summary(limit=2000)

    trend_dist = t_df["overall_trend"].value_counts().to_dict() if not t_df.empty else {}
    trend_map = (
        t_df.set_index("equipment_id")["overall_trend"].to_dict()
        if not t_df.empty
        else {}
    )
    for row in equipment_rows:
        row["overall_trend"] = trend_map.get(row["equipment_id"], "Stable")

    # Include first 20 tracked units with multi-point history
    tracked_20 = []
    for i in range(1, 21):
        eq_id = f"OBS-{i:06d}"
        tr = analyse_trends(eq_id)
        tracked_20.append(
            {
                "equipment_id": eq_id,
                "overall_trend": tr["overall_trend"],
                "sensor_trend": tr["sensor_trend"],
                "failure_probability_trend": tr["failure_probability_trend"],
                "health_score_trend": tr["health_score_trend"],
                "latest_health_score": tr["latest_health_score"],
                "health_score_delta": tr["health_score_delta"],
                "latest_failure_probability": tr["latest_failure_probability"],
            }
        )

    alerts_list = get_alerts(limit=160)

    return {
        "backend": backend,
        "table_counts": counts,
        "kpis": {
            "total_equipment": counts.get("equipment", 0),
            "avg_health_score": avg_health,
            "avg_failure_probability": avg_prob,
            "good_health_count": health_dist.get("Good", 0),
            "moderate_health_count": health_dist.get("Moderate", 0),
            "poor_health_count": health_dist.get("Poor", 0),
            "critical_risk_count": risk_dist.get("Critical", 0),
            "warning_risk_count": risk_dist.get("Warning", 0),
            "healthy_risk_count": risk_dist.get("Healthy", 0),
            "total_alerts": len(alerts_list),
            "active_alerts": sum(1 for a in alerts_list if a["status"] in ("New", "Acknowledged")),
        },
        "health_dist": health_dist,
        "risk_dist": risk_dist,
        "issue_dist": issue_dist,
        "trend_dist": {k: int(v) for k, v in trend_dist.items()},
        "equipment_rows": equipment_rows,
        "tracked_trends": tracked_20,
        "alerts": alerts_list,
    }


def generate_dashboard_overview_png(payload: Dict[str, Any]) -> Path:
    """Create a 4-panel visual summary chart in reports/member5_dashboard_overview.png."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=140)
    fig.patch.set_facecolor("#0f172a")
    for ax in axes.flat:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#e2e8f0")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # Panel 1: Equipment Health Status (Member 5 Part C)
    ax1 = axes[0, 0]
    h_labels = ["Good (80-100)", "Moderate (50-79)", "Poor (<50)"]
    h_vals = [
        payload["health_dist"].get("Good", 0),
        payload["health_dist"].get("Moderate", 0),
        payload["health_dist"].get("Poor", 0),
    ]
    h_colors = ["#10b981", "#f59e0b", "#ef4444"]
    bars1 = ax1.bar(h_labels, h_vals, color=h_colors, width=0.55)
    ax1.set_title("Equipment Health Score Classification (Member 5)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Equipment Count", color="#cbd5e1")
    for b in bars1:
        ax1.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 25,
            f"{int(b.get_height()):,}",
            ha="center",
            color="#f8fafc",
            fontweight="bold",
        )
    ax1.set_ylim(0, max(h_vals) * 1.15 if h_vals else 100)

    # Panel 2: Risk Level Distribution (Member 2)
    ax2 = axes[0, 1]
    r_labels = ["Healthy", "Warning", "Critical"]
    r_vals = [
        payload["risk_dist"].get("Healthy", 0),
        payload["risk_dist"].get("Warning", 0),
        payload["risk_dist"].get("Critical", 0),
    ]
    r_colors = ["#22c55e", "#eab308", "#f43f5e"]
    bars2 = ax2.bar(r_labels, r_vals, color=r_colors, width=0.55)
    ax2.set_title("Equipment Risk Level Classification (Member 2)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Equipment Count", color="#cbd5e1")
    for b in bars2:
        ax2.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 25,
            f"{int(b.get_height()):,}",
            ha="center",
            color="#f8fafc",
            fontweight="bold",
        )
    ax2.set_ylim(0, max(r_vals) * 1.15 if r_vals else 100)

    # Panel 3: Historical Trend Analysis (Member 5 Part D)
    ax3 = axes[1, 0]
    t_labels = ["Stable", "Deteriorating", "Improving"]
    t_vals = [
        payload["trend_dist"].get("Stable", 0),
        payload["trend_dist"].get("Deteriorating", 0),
        payload["trend_dist"].get("Improving", 0),
    ]
    t_colors = ["#38bdf8", "#ef4444", "#10b981"]
    bars3 = ax3.bar(t_labels, t_vals, color=t_colors, width=0.55)
    ax3.set_title("Historical Trend Classification (Member 5)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Equipment Count", color="#cbd5e1")
    for b in bars3:
        ax3.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 25,
            f"{int(b.get_height()):,}",
            ha="center",
            color="#f8fafc",
            fontweight="bold",
        )
    ax3.set_ylim(0, max(t_vals) * 1.15 if t_vals else 100)

    # Panel 4: Database Table Row Counts (Member 5 Part A & B)
    ax4 = axes[1, 1]
    tbl_names = list(payload["table_counts"].keys())
    tbl_vals = list(payload["table_counts"].values())
    bars4 = ax4.barh(tbl_names, tbl_vals, color="#6366f1")
    ax4.set_title("Shared Database Table Records (Member 5 DB)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Stored SQL Rows", color="#cbd5e1")
    for b in bars4:
        ax4.text(
            b.get_width() + 30,
            b.get_y() + b.get_height() / 2,
            f"{int(b.get_width()):,}",
            va="center",
            color="#f8fafc",
            fontweight="bold",
        )
    ax4.set_xlim(0, max(tbl_vals) * 1.18 if tbl_vals else 100)

    fig.suptitle(
        f"Milestone 3 – Predictive Maintenance, Health Scoring & Trend Analysis Dashboard "
        f"(Avg Health: {payload['kpis']['avg_health_score']}/100 | Active Alerts: {payload['kpis']['active_alerts']})",
        color="#ffffff",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(DASHBOARD_PNG_PATH, facecolor=fig.get_facecolor())
    plt.close(fig)
    return DASHBOARD_PNG_PATH


def render_dashboard_html(payload: Dict[str, Any]) -> str:
    """Render the interactive HTML dashboard with embedded data and live API support."""
    data_json = json.dumps(payload)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Predictive Maintenance & Health Scoring Dashboard (Milestone 3)</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #0b1120;
      --panel: #151f32;
      --panel-hover: #1e293b;
      --border: #26354d;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --good: #10b981;
      --warn: #f59e0b;
      --crit: #ef4444;
      --indigo: #6366f1;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding: 20px 28px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
      padding-bottom: 18px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 22px;
    }}
    .title-group h1 {{
      font-size: 1.55rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }}
    .title-group p {{
      color: var(--muted);
      font-size: 0.9rem;
      margin-top: 4px;
    }}
    .db-badge {{
      background: rgba(16, 185, 129, 0.14);
      border: 1px solid rgba(16, 185, 129, 0.4);
      color: #34d399;
      padding: 8px 14px;
      border-radius: 999px;
      font-size: 0.84rem;
      font-weight: 600;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(195px, 1fr));
      gap: 16px;
      margin-bottom: 22px;
    }}
    .kpi-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px 18px;
    }}
    .kpi-label {{
      color: var(--muted);
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .kpi-value {{
      font-size: 1.85rem;
      font-weight: 700;
      margin: 6px 0 4px;
    }}
    .kpi-sub {{
      font-size: 0.8rem;
      color: var(--muted);
    }}
    .tabs {{
      display: flex;
      gap: 10px;
      margin-bottom: 20px;
      flex-wrap: wrap;
    }}
    .tab-btn {{
      background: var(--panel);
      border: 1px solid var(--border);
      color: var(--muted);
      padding: 10px 18px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
      font-size: 0.9rem;
      transition: all 0.15s;
    }}
    .tab-btn.active, .tab-btn:hover {{
      background: var(--indigo);
      border-color: var(--indigo);
      color: #fff;
    }}
    .tab-content {{ display: none; }}
    .tab-content.active {{ display: block; }}
    .charts-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 18px;
      margin-bottom: 24px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px;
    }}
    .card h3 {{
      font-size: 1rem;
      margin-bottom: 12px;
      color: #e2e8f0;
    }}
    .chart-box {{
      position: relative;
      height: 250px;
    }}
    .filter-bar {{
      display: flex;
      gap: 12px;
      margin-bottom: 14px;
      flex-wrap: wrap;
    }}
    input, select, button.action-btn {{
      background: #0f172a;
      border: 1px solid var(--border);
      color: var(--text);
      padding: 8px 12px;
      border-radius: 8px;
      font-size: 0.88rem;
    }}
    button.action-btn {{
      background: var(--indigo);
      border-color: var(--indigo);
      cursor: pointer;
      font-weight: 600;
    }}
    button.action-btn:hover {{ opacity: 0.92; }}
    .table-wrap {{
      overflow-x: auto;
      max-height: 520px;
      overflow-y: auto;
      border: 1px solid var(--border);
      border-radius: 10px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }}
    th, td {{
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }}
    th {{
      background: #0f172a;
      color: var(--muted);
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    tr:hover {{ background: var(--panel-hover); }}
    .pill {{
      display: inline-block;
      padding: 3px 9px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 700;
    }}
    .pill.Good, .pill.Healthy, .pill.Improving, .pill.Resolved {{
      background: rgba(16, 185, 129, 0.16);
      color: #34d399;
    }}
    .pill.Moderate, .pill.Warning, .pill.Acknowledged {{
      background: rgba(245, 158, 11, 0.16);
      color: #fbbf24;
    }}
    .pill.Poor, .pill.Critical, .pill.Deteriorating, .pill.New {{
      background: rgba(239, 68, 68, 0.18);
      color: #f87171;
    }}
    .pill.Stable {{
      background: rgba(56, 189, 248, 0.16);
      color: #38bdf8;
    }}
    .sim-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-bottom: 16px;
    }}
    .sim-field label {{
      display: block;
      font-size: 0.8rem;
      color: var(--muted);
      margin-bottom: 5px;
    }}
    .sim-field input, .sim-field select {{ width: 100%; }}
  </style>
</head>
<body>
  <header>
    <div class="title-group">
      <h1>Early Equipment Failure Detection & Health Analytics</h1>
      <p>Milestone 3 Integrated System — Members 1, 2, 3, 4 & Member 5 (Database, Health Scoring & Trend Analysis)</p>
    </div>
    <div class="db-badge" id="dbBadge">Connected to SQL Database</div>
  </header>

  <div class="kpi-grid" id="kpiGrid"></div>

  <div class="tabs">
    <button class="tab-btn active" onclick="switchTab('overview', this)">1. Fleet Overview & Charts</button>
    <button class="tab-btn" onclick="switchTab('equipment', this)">2. Equipment Health & Risk Explorer</button>
    <button class="tab-btn" onclick="switchTab('trends', this)">3. Member 5 Historical Trends</button>
    <button class="tab-btn" onclick="switchTab('alerts', this)">4. Alert Management (160 Alerts)</button>
    <button class="tab-btn" onclick="switchTab('database', this)">5. Database Tables & Live Simulator</button>
    <button class="tab-btn" onclick="switchTab('entry', this)">6. Manual Sensor Entry</button>
  </div>

  <!-- TAB 1: OVERVIEW -->
  <div id="tab-overview" class="tab-content active">
    <div class="charts-grid">
      <div class="card">
        <h3>Equipment Health Score Classification (Member 5 Part C)</h3>
        <div class="chart-box"><canvas id="healthChart"></canvas></div>
      </div>
      <div class="card">
        <h3>Equipment Risk Level Distribution (Member 2)</h3>
        <div class="chart-box"><canvas id="riskChart"></canvas></div>
      </div>
      <div class="card">
        <h3>Failure Category Breakdown (Member 3 Rules)</h3>
        <div class="chart-box"><canvas id="issueChart"></canvas></div>
      </div>
      <div class="card">
        <h3>Shared SQL Database Table Records (Member 5 Part A/B)</h3>
        <div class="chart-box"><canvas id="dbChart"></canvas></div>
      </div>
    </div>
  </div>

  <!-- TAB 2: EQUIPMENT EXPLORER -->
  <div id="tab-equipment" class="tab-content">
    <div class="card">
      <h3>Equipment Health Scoring, Anomaly & Maintenance Table</h3>
      <div class="filter-bar">
        <input type="text" id="eqSearch" placeholder="Search Equipment ID (e.g. OBS-000001)..." oninput="renderEquipmentTable()" />
        <select id="healthFilter" onchange="renderEquipmentTable()">
          <option value="ALL">All Health Statuses</option>
          <option value="Poor">Poor (&lt;50)</option>
          <option value="Moderate">Moderate (50-79)</option>
          <option value="Good">Good (80-100)</option>
        </select>
        <select id="riskFilter" onchange="renderEquipmentTable()">
          <option value="ALL">All Risk Levels</option>
          <option value="Critical">Critical</option>
          <option value="Warning">Warning</option>
          <option value="Healthy">Healthy</option>
        </select>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Equipment ID</th>
              <th>Type</th>
              <th>Health Score</th>
              <th>Health Status</th>
              <th>Failure Prob</th>
              <th>Risk Level</th>
              <th>Trend</th>
              <th>Abnormal Sensors</th>
              <th>Primary Issue (M3)</th>
              <th>Recommended Action</th>
            </tr>
          </thead>
          <tbody id="eqTableBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- TAB 3: TRENDS -->
  <div id="tab-trends" class="tab-content">
    <div class="card">
      <h3>Member 5 Part D – Historical Trend Analysis (7-Day Multi-Point Tracked Fleet)</h3>
      <p style="color:var(--muted); font-size:0.86rem; margin-bottom:14px;">
        Compares historical vs. current sensor readings, ML failure probability, health score, and alert status stored in the SQL database.
      </p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Equipment ID</th>
              <th>Overall Trend</th>
              <th>Sensor Trend</th>
              <th>Failure Prob Trend</th>
              <th>Health Score Trend</th>
              <th>Latest Health Score</th>
              <th>7-Day Health Score &Delta;</th>
              <th>Latest Failure Prob</th>
            </tr>
          </thead>
          <tbody id="trendTableBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- TAB 4: ALERTS -->
  <div id="tab-alerts" class="tab-content">
    <div class="card">
      <h3>Automated Equipment Alerts (Member 4 + Member 5 Database Storage)</h3>
      <div class="filter-bar">
        <select id="alertRiskFilter" onchange="renderAlertsTable()">
          <option value="ALL">All Alert Severities</option>
          <option value="Critical">Critical</option>
          <option value="Warning">Warning</option>
        </select>
        <select id="alertStatusFilter" onchange="renderAlertsTable()">
          <option value="ALL">All Statuses</option>
          <option value="New">New</option>
          <option value="Acknowledged">Acknowledged</option>
          <option value="Resolved">Resolved</option>
        </select>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Equipment ID</th>
              <th>Risk Level</th>
              <th>Failure Prob</th>
              <th>Condition / Issue</th>
              <th>Status</th>
              <th>Description</th>
            </tr>
          </thead>
          <tbody id="alertsTableBody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- TAB 5: DATABASE & LIVE SIMULATOR -->
  <div id="tab-database" class="tab-content">
    <div class="card" style="margin-bottom:20px;">
      <h3>Interactive Live Equipment Health & Failure Calculator (Formula Verification)</h3>
      <p style="color:var(--muted); font-size:0.86rem; margin-bottom:14px;">
        Test Member 5's health scoring formula <code>Health Score = 100 * (1 - Failure Probability)</code>, Member 3's physical failure rules, and Member 2's risk classification in real time:
      </p>
      <div class="sim-grid">
        <div class="sim-field"><label>Equipment ID</label><input id="simEqId" value="EQ-LIVE-01" /></div>
        <div class="sim-field"><label>Equipment Type</label>
          <select id="simType"><option value="L">L (Low)</option><option value="M">M (Medium)</option><option value="H">H (High)</option></select>
        </div>
        <div class="sim-field"><label>Air Temp [K]</label><input type="number" step="0.1" id="simAir" value="302.5" /></div>
        <div class="sim-field"><label>Process Temp [K]</label><input type="number" step="0.1" id="simProc" value="309.0" /></div>
        <div class="sim-field"><label>Rotational Speed [rpm]</label><input type="number" step="1" id="simSpeed" value="1320" /></div>
        <div class="sim-field"><label>Torque [Nm]</label><input type="number" step="0.5" id="simTorque" value="62.5" /></div>
        <div class="sim-field"><label>Tool Wear [min]</label><input type="number" step="1" id="simWear" value="215" /></div>
      </div>
      <button class="action-btn" onclick="runSimulator()">Evaluate Equipment Health & Maintenance</button>
      <div id="simResult" style="margin-top:16px;"></div>
    </div>
  </div>

  <script>
    const DATA = {data_json};

    function initDashboard() {{
      const k = DATA.kpis;
      document.getElementById('dbBadge').textContent =
        `SQL Database Active (${{DATA.backend.toUpperCase()}}) • ${{Object.values(DATA.table_counts).reduce((a,b)=>a+b, 0).toLocaleString()}} Total Records`;

      const kpis = [
        {{ label: 'Total Monitored Equipment', value: k.total_equipment.toLocaleString(), sub: 'Stored in equipment table', color: '#38bdf8' }},
        {{ label: 'Fleet Avg Health Score', value: `${{k.avg_health_score}} / 100`, sub: `Good: ${{k.good_health_count}} | Mod: ${{k.moderate_health_count}} | Poor: ${{k.poor_health_count}}`, color: '#10b981' }},
        {{ label: 'Poor Health (<50 Score)', value: k.poor_health_count, sub: 'High failure probability units', color: '#ef4444' }},
        {{ label: 'Critical & Warning Risk', value: `${{k.critical_risk_count}} / ${{k.warning_risk_count}}`, sub: `${{k.healthy_risk_count}} Healthy units`, color: '#f59e0b' }},
        {{ label: 'Active Alerts (M4)', value: k.active_alerts, sub: `Deduplicated in alerts table`, color: '#a855f7' }}
      ];
      document.getElementById('kpiGrid').innerHTML = kpis.map(item => `
        <div class="kpi-card">
          <div class="kpi-label">${{item.label}}</div>
          <div class="kpi-value" style="color:${{item.color}}">${{item.value}}</div>
          <div class="kpi-sub">${{item.sub}}</div>
        </div>
      `).join('');

      new Chart(document.getElementById('healthChart'), {{
        type: 'doughnut',
        data: {{
          labels: ['Good (80-100)', 'Moderate (50-79)', 'Poor (<50)'],
          datasets: [{{
            data: [DATA.health_dist.Good || 0, DATA.health_dist.Moderate || 0, DATA.health_dist.Poor || 0],
            backgroundColor: ['#10b981', '#f59e0b', '#ef4444'],
            borderWidth: 0
          }}]
        }},
        options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ labels: {{ color: '#e2e8f0' }} }} }} }}
      }});

      new Chart(document.getElementById('riskChart'), {{
        type: 'bar',
        data: {{
          labels: ['Healthy', 'Warning', 'Critical'],
          datasets: [{{
            label: 'Equipment Count',
            data: [DATA.risk_dist.Healthy || 0, DATA.risk_dist.Warning || 0, DATA.risk_dist.Critical || 0],
            backgroundColor: ['#22c55e', '#eab308', '#f43f5e'],
            borderRadius: 6
          }}]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{ x: {{ ticks: {{ color: '#cbd5e1' }} }}, y: {{ ticks: {{ color: '#cbd5e1' }} }} }}
        }}
      }});

      new Chart(document.getElementById('issueChart'), {{
        type: 'bar',
        data: {{
          labels: Object.keys(DATA.issue_dist),
          datasets: [{{
            label: 'Detected Cases',
            data: Object.values(DATA.issue_dist),
            backgroundColor: '#38bdf8',
            borderRadius: 6
          }}]
        }},
        options: {{
          indexAxis: 'y',
          responsive: true, maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{ x: {{ ticks: {{ color: '#cbd5e1' }} }}, y: {{ ticks: {{ color: '#cbd5e1' }} }} }}
        }}
      }});

      new Chart(document.getElementById('dbChart'), {{
        type: 'bar',
        data: {{
          labels: Object.keys(DATA.table_counts),
          datasets: [{{
            label: 'Rows in SQL Table',
            data: Object.values(DATA.table_counts),
            backgroundColor: '#6366f1',
            borderRadius: 6
          }}]
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{ legend: {{ display: false }} }},
          scales: {{ x: {{ ticks: {{ color: '#cbd5e1' }} }}, y: {{ ticks: {{ color: '#cbd5e1' }} }} }}
        }}
      }});

      renderEquipmentTable();
      renderTrendTable();
      renderAlertsTable();
      runSimulator();
    }}

    function switchTab(tabId, btn) {{
      document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
      document.getElementById('tab-' + tabId).classList.add('active');
      btn.classList.add('active');
    }}

    function renderEquipmentTable() {{
      const q = document.getElementById('eqSearch').value.trim().toLowerCase();
      const hf = document.getElementById('healthFilter').value;
      const rf = document.getElementById('riskFilter').value;
      const rows = DATA.equipment_rows.filter(r => {{
        if (q && !r.equipment_id.toLowerCase().includes(q) && !(r.primary_issue || '').toLowerCase().includes(q)) return false;
        if (hf !== 'ALL' && r.health_status !== hf) return false;
        if (rf !== 'ALL' && r.risk_level !== rf) return false;
        return true;
      }});
      document.getElementById('eqTableBody').innerHTML = rows.map(r => `
        <tr>
          <td><strong>${{r.equipment_id}}</strong></td>
          <td>${{r.equipment_type}}</td>
          <td><strong>${{Number(r.health_score).toFixed(2)}}</strong></td>
          <td><span class="pill ${{r.health_status}}">${{r.health_status}}</span></td>
          <td>${{(Number(r.failure_probability) * 100).toFixed(2)}}%</td>
          <td><span class="pill ${{r.risk_level}}">${{r.risk_level}}</span></td>
          <td><span class="pill ${{r.overall_trend}}">${{r.overall_trend}}</span></td>
          <td>${{r.abnormal_sensors}}</td>
          <td>${{r.primary_issue || 'Normal'}}</td>
          <td>${{r.action_type || 'Routine Monitoring'}}</td>
        </tr>
      `).join('');
    }}

    function renderTrendTable() {{
      document.getElementById('trendTableBody').innerHTML = DATA.tracked_trends.map(t => `
        <tr>
          <td><strong>${{t.equipment_id}}</strong></td>
          <td><span class="pill ${{t.overall_trend}}">${{t.overall_trend}}</span></td>
          <td><span class="pill ${{t.sensor_trend}}">${{t.sensor_trend}}</span></td>
          <td><span class="pill ${{t.failure_probability_trend}}">${{t.failure_probability_trend}}</span></td>
          <td><span class="pill ${{t.health_score_trend}}">${{t.health_score_trend}}</span></td>
          <td><strong>${{Number(t.latest_health_score).toFixed(2)}}</strong></td>
          <td style="color:${{t.health_score_delta >= 0 ? '#34d399' : '#f87171'}}">
            ${{t.health_score_delta >= 0 ? '+' : ''}}${{Number(t.health_score_delta).toFixed(2)}}
          </td>
          <td>${{(Number(t.latest_failure_probability) * 100).toFixed(2)}}%</td>
        </tr>
      `).join('');
    }}

    function renderAlertsTable() {{
      const rf = document.getElementById('alertRiskFilter').value;
      const sf = document.getElementById('alertStatusFilter').value;
      const rows = DATA.alerts.filter(a => {{
        if (rf !== 'ALL' && a.risk_level !== rf) return false;
        if (sf !== 'ALL' && a.status !== sf) return false;
        return true;
      }});
      document.getElementById('alertsTableBody').innerHTML = rows.map(a => `
        <tr>
          <td><strong>${{a.equipment_id}}</strong></td>
          <td><span class="pill ${{a.risk_level}}">${{a.risk_level}}</span></td>
          <td>${{(Number(a.failure_probability) * 100).toFixed(2)}}%</td>
          <td>${{a.condition}}</td>
          <td><span class="pill ${{a.status}}">${{a.status}}</span></td>
          <td style="max-width:480px;">${{a.description}}</td>
        </tr>
      `).join('');
    }}

    function runSimulator() {{
      const eqId = document.getElementById('simEqId').value;
      const type = document.getElementById('simType').value;
      const air = parseFloat(document.getElementById('simAir').value);
      const proc = parseFloat(document.getElementById('simProc').value);
      const speed = parseFloat(document.getElementById('simSpeed').value);
      const torque = parseFloat(document.getElementById('simTorque').value);
      const wear = parseFloat(document.getElementById('simWear').value);

      const tempDiff = proc - air;
      const powerW = torque * (speed * 2 * Math.PI / 60);
      const strain = wear * torque;
      const osfLimits = {{ L: 11000, M: 12000, H: 13000 }};

      const issues = [];
      if (wear >= 200) issues.push('Tool Wear Failure (TWF)');
      else if (wear >= 160) issues.push('High Tool Wear Warning');
      if (tempDiff < 8.6 && speed < 1380) issues.push('Heat Dissipation Failure (HDF)');
      if (powerW < 3500 || powerW > 9000) issues.push('Power Failure (PWF)');
      if (strain > osfLimits[type]) issues.push('Overstrain Failure (OSF)');

      // Estimate failure probability from violated physical rules or normal baseline
      let prob = 0.03;
      if (issues.length >= 2) prob = 0.92;
      else if (issues.length === 1) prob = issues[0].includes('Warning') ? 0.45 : 0.84;
      else if (torque > 55 || wear > 140) prob = 0.35;

      const healthScore = (100 * (1 - prob)).toFixed(2);
      const healthStatus = healthScore >= 80 ? 'Good' : (healthScore >= 50 ? 'Moderate' : 'Poor');
      const riskLevel = prob >= 0.70 || issues.length >= 2 ? 'Critical' : (prob >= 0.30 || issues.length === 1 ? 'Warning' : 'Healthy');

      document.getElementById('simResult').innerHTML = `
        <div class="kpi-grid" style="margin-bottom:0;">
          <div class="kpi-card">
            <div class="kpi-label">Equipment ID</div>
            <div class="kpi-value">${{eqId}} (${{type}})</div>
            <div class="kpi-sub">Power: ${{powerW.toFixed(0)}} W | &Delta;T: ${{tempDiff.toFixed(1)}} K</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Health Score (Member 5)</div>
            <div class="kpi-value">${{healthScore}} / 100</div>
            <div class="kpi-sub">Status: <span class="pill ${{healthStatus}}">${{healthStatus}}</span></div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Failure Prob & Risk</div>
            <div class="kpi-value">${{(prob * 100).toFixed(1)}}%</div>
            <div class="kpi-sub">Risk: <span class="pill ${{riskLevel}}">${{riskLevel}}</span></div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Identified Failure Modes (M3)</div>
            <div class="kpi-value" style="font-size:1.05rem;">${{issues.length ? issues.join(', ') : 'Normal Operation'}}</div>
            <div class="kpi-sub">Strain: ${{strain.toFixed(0)}} min·Nm</div>
          </div>
        </div>
      `;
    }}

    initDashboard();
  </script>
</body>
</html>"""


def build_static_dashboard_artifacts() -> Dict[str, Any]:
    """Generate both the standalone HTML dashboard and overview PNG chart."""
    payload = fetch_dashboard_payload()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html_str = render_dashboard_html(payload)
    DASHBOARD_HTML_PATH.write_text(html_str, encoding="utf-8")
    generate_dashboard_overview_png(payload)
    return payload


def wsgi_app(environ, start_response):
    """Lightweight WSGI server for live interactive browsing at http://localhost:8050."""
    path = environ.get("PATH_INFO", "/")
    if path in ("/", "/index.html"):
        payload = fetch_dashboard_payload()
        body = render_dashboard_html(payload).encode("utf-8")
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [body]
    if path == "/api/summary":
        payload = fetch_dashboard_payload()
        body = json.dumps(payload).encode("utf-8")
        start_response("200 OK", [("Content-Type", "application/json")])
        return [body]
    if path == "/api/insert_sensor" and environ.get("REQUEST_METHOD", "GET") == "POST":
        # Read JSON body
        try:
            length = int(environ.get("CONTENT_LENGTH", "0"))
        except ValueError:
            length = 0
        body = environ["wsgi.input"].read(length) if length > 0 else b""
        try:
            data = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            start_response("400 Bad Request", [("Content-Type", "application/json")])
            return [json.dumps({"error":"Invalid JSON"}).encode("utf-8")]
        equipment_id = data.get("equipment_id")
        if not equipment_id:
            start_response("400 Bad Request", [("Content-Type", "application/json")])
            return [json.dumps({"error":"Missing equipment_id"}).encode("utf-8")]
        # Extract other fields with defaults
        air_temp = float(data.get("air_temperature", 300.0))
        process_temp = float(data.get("process_temperature", 310.0))
        rotational_speed = float(data.get("rotational_speed", 1500.0))
        torque = float(data.get("torque", 40.0))
        tool_wear = float(data.get("tool_wear", 0.0))
        equipment_type = data.get("equipment_type", "L")
        product_id = data.get("product_id")
        # Insert into DB
        insert_sensor_reading(
            equipment_id=equipment_id,
            air_temp=air_temp,
            process_temp=process_temp,
            rotational_speed=rotational_speed,
            torque=torque,
            tool_wear=tool_wear,
            equipment_type=equipment_type,
            product_id=product_id,
        )
        start_response("200 OK", [("Content-Type", "application/json")])
        return [json.dumps({"status":"success"}).encode("utf-8")]

    if path == "/manual_entry":
        # Serve the manual entry HTML form
        try:
            with open(os.path.join(PROJECT_DIR, "src", "manual_entry_form.html"), "r", encoding="utf-8") as f:
                html = f.read()
        except Exception as e:
            start_response("500 Internal Server Error", [("Content-Type", "text/plain")])
            return [str(e).encode("utf-8")]
        start_response("200 OK", [("Content-Type", "text/html; charset=utf-8")])
        return [html.encode("utf-8")]
    start_response("404 Not Found", [("Content-Type", "text/plain")])
    return [b"Not Found"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Milestone 3 Predictive Maintenance Dashboard")
    parser.add_argument("--export-only", action="store_true", help="Export HTML and PNG without starting server")
    parser.add_argument("--port", type=int, default=8050, help="Port to run live dashboard server")
    args = parser.parse_args()

    payload_data = build_static_dashboard_artifacts()
    print(f"Dashboard HTML saved to: {DASHBOARD_HTML_PATH}")
    print(f"Dashboard PNG saved to : {DASHBOARD_PNG_PATH}")

    if not args.export_only:
        print(f"\nStarting live dashboard server at http://localhost:{args.port} ...")
        with make_server("127.0.0.1", args.port, wsgi_app) as httpd:
            httpd.serve_forever()
