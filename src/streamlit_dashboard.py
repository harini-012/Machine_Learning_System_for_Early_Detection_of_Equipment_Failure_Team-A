"""
Streamlit Interactive Dashboard – Equipment Failure Prediction & Health Analytics
(Milestone 3 & Milestone 4 Web Dashboard)

Run with:
    streamlit run src/streamlit_dashboard.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is on sys.path when invoked via `streamlit run src/streamlit_dashboard.py`
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.dashboard_app import (
    evaluate_manual_sensor_input,
    fetch_dashboard_payload,
    IQR_LIMITS,
)
from src.database import (
    update_alert_status_in_db,
    retrieve_equipment_history,
    get_database_backend,
)
from src.trend_analysis import analyse_trends

st.set_page_config(
    page_title="Equipment Failure Prediction & Health Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRESET_SCENARIOS = {
    "Custom Manual Entry": {
        "eq_id": "EQ-MANUAL-01",
        "type": "L",
        "air": 300.5,
        "proc": 310.2,
        "speed": 1320.0,
        "torque": 62.5,
        "wear": 215.0,
    },
    "Healthy Equipment Baseline": {
        "eq_id": "EQ-1001",
        "type": "M",
        "air": 298.1,
        "proc": 308.6,
        "speed": 1551.0,
        "torque": 42.8,
        "wear": 45.0,
    },
    "Tool Wear Failure (TWF)": {
        "eq_id": "EQ-1002",
        "type": "L",
        "air": 298.5,
        "proc": 308.7,
        "speed": 1400.0,
        "torque": 45.0,
        "wear": 215.0,
    },
    "Heat Dissipation Failure (HDF)": {
        "eq_id": "EQ-1003",
        "type": "L",
        "air": 302.5,
        "proc": 309.0,
        "speed": 1320.0,
        "torque": 62.5,
        "wear": 110.0,
    },
    "Power Failure (PWF - Overload)": {
        "eq_id": "EQ-1004",
        "type": "H",
        "air": 300.0,
        "proc": 310.0,
        "speed": 2600.0,
        "torque": 68.0,
        "wear": 30.0,
    },
    "Overstrain Failure (OSF)": {
        "eq_id": "EQ-1005",
        "type": "L",
        "air": 300.0,
        "proc": 310.0,
        "speed": 1420.0,
        "torque": 68.0,
        "wear": 190.0,
    },
}


def render_header_and_kpis(payload: dict) -> None:
    st.title("⚙️ ML System for Early Detection of Equipment Failure")
    st.caption(
        f"**Integrated Milestone 3 & 4 Dashboard** • Connected to Shared SQL Database "
        f"(`{payload['backend'].upper()}`) • **{sum(payload['table_counts'].values()):,}** Total Stored Records"
    )

    kpis = payload["kpis"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Monitored Equipment", f"{kpis['total_equipment']:,}", "SQL equipment table")
    c2.metric(
        "Fleet Avg Health Score",
        f"{kpis['avg_health_score']:.2f} / 100",
        f"{kpis['good_health_count']:,} Good",
    )
    c3.metric(
        "Poor Health (<50)",
        f"{kpis['poor_health_count']:,}",
        f"{kpis['moderate_health_count']:,} Moderate",
        delta_color="inverse",
    )
    c4.metric(
        "Critical Risk Units",
        f"{kpis['critical_risk_count']:,}",
        f"{kpis['warning_risk_count']:,} Warning",
        delta_color="inverse",
    )
    c5.metric(
        "Active Alerts (M4)",
        f"{kpis['active_alerts']:,}",
        f"{kpis['total_alerts']:,} Total Alerts",
        delta_color="inverse",
    )


def render_manual_prediction_tab() -> None:
    st.subheader("🧪 Manual Sensor Input Form & Real-Time Failure Prediction")
    st.markdown(
        "Enter equipment sensor readings manually (or load a preset test scenario) to run "
        "**Member 1 (XGBoost ML Prediction)**, **Member 2 (Anomaly & Risk Classification)**, "
        "**Member 3 (Failure Mode & Maintenance)**, **Member 4 (Alert Generation)**, and "
        "**Member 5 (Health Scoring, SQL Database Storage & Trend Analysis)**."
    )

    col_form, col_result = st.columns([1.05, 1.35], gap="large")

    with col_form:
        preset_name = st.selectbox(
            "Load Test Scenario Preset",
            options=list(PRESET_SCENARIOS.keys()),
            index=0,
        )
        preset = PRESET_SCENARIOS[preset_name]

        with st.form("manual_sensor_input_form", clear_on_submit=False):
            st.markdown("#### Equipment & Sensor Parameters")
            id_col, type_col = st.columns(2)
            with id_col:
                equipment_id = st.text_input("Equipment ID", value=preset["eq_id"])
            with type_col:
                type_options = ["L", "M", "H"]
                equipment_type = st.selectbox(
                    "Equipment Type (Quality Variant)",
                    options=type_options,
                    index=type_options.index(preset["type"]),
                )

            air_temp = st.number_input(
                "1. Air Temperature [K] (Normal IQR: 293.5 – 306.3 K)",
                min_value=250.0,
                max_value=350.0,
                value=float(preset["air"]),
                step=0.1,
                format="%.1f",
            )
            proc_temp = st.number_input(
                "2. Process Temperature [K] (Normal IQR: 305.35 – 314.55 K)",
                min_value=250.0,
                max_value=360.0,
                value=float(preset["proc"]),
                step=0.1,
                format="%.1f",
            )
            rot_speed = st.number_input(
                "3. Rotational Speed [rpm] (Normal IQR: 1150 – 1894 rpm)",
                min_value=500.0,
                max_value=4000.0,
                value=float(preset["speed"]),
                step=10.0,
                format="%.0f",
            )
            torque = st.number_input(
                "4. Torque [Nm] (Normal IQR: 13.3 – 66.1 Nm)",
                min_value=0.0,
                max_value=150.0,
                value=float(preset["torque"]),
                step=0.5,
                format="%.1f",
            )
            tool_wear = st.number_input(
                "5. Tool Wear [min] (Warning ≥ 160 min | Critical ≥ 200 min)",
                min_value=0.0,
                max_value=500.0,
                value=float(preset["wear"]),
                step=1.0,
                format="%.0f",
            )

            save_to_db = st.checkbox(
                "Save prediction, anomalies, maintenance & health score to SQL Database",
                value=True,
            )

            submitted = st.form_submit_button(
                "🔍 Predict Failure Status & Compute Health Score",
                use_container_width=True,
                type="primary",
            )

    if submitted or "last_prediction_result" not in st.session_state:
        result = evaluate_manual_sensor_input(
            {
                "equipment_id": equipment_id,
                "equipment_type": equipment_type,
                "air_temperature": air_temp,
                "process_temperature": proc_temp,
                "rotational_speed": rot_speed,
                "torque": torque,
                "tool_wear": tool_wear,
                "save_to_db": save_to_db if submitted else False,
            }
        )
        st.session_state["last_prediction_result"] = result
        if submitted:
            history_log = st.session_state.setdefault("manual_submission_log", [])
            history_log.insert(0, result)

    res = st.session_state["last_prediction_result"]
    m1 = res["member1_prediction"]
    m2 = res["member2_anomaly"]
    m3 = res["member3_maintenance"]
    m4 = res["member4_alert"]
    m5 = res["member5_health_and_trend"]

    with col_result:
        st.markdown("#### Predicted Equipment Failure Status")

        # Prominent Failure Status Banner
        if m1["predicted_failure"] == 1 or m2["risk_level"] == "Critical":
            st.error(
                f"🚨 **PREDICTED STATUS: {m1['prediction_label'].upper()}**  |  "
                f"**Risk Level: {m2['risk_level'].upper()}**  |  "
                f"**Failure Probability: {m1['failure_probability']*100:.2f}%**  |  "
                f"**Health Score: {m5['health_score']:.2f}/100 ({m5['health_status']})**"
            )
        elif m2["risk_level"] == "Warning":
            st.warning(
                f"⚠️ **PREDICTED STATUS: {m1['prediction_label'].upper()} (WARNING RISK)**  |  "
                f"**Failure Probability: {m1['failure_probability']*100:.2f}%**  |  "
                f"**Health Score: {m5['health_score']:.2f}/100 ({m5['health_status']})**"
            )
        else:
            st.success(
                f"✅ **PREDICTED STATUS: {m1['prediction_label'].upper()} (HEALTHY)**  |  "
                f"**Failure Probability: {m1['failure_probability']*100:.2f}%**  |  "
                f"**Health Score: {m5['health_score']:.2f}/100 ({m5['health_status']})**"
            )

        # 4 Key Metric Cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("M1 Prediction", m1["prediction_label"], f"Prob: {m1['failure_probability']*100:.2f}%")
        k2.metric("M5 Health Score", f"{m5['health_score']:.2f} / 100", f"Status: {m5['health_status']}")
        k3.metric("M2 Risk Level", m2["risk_level"], f"{m2['anomaly_count']} Abnormal Sensor(s)")
        k4.metric("M5 Overall Trend", m5["overall_trend"], f"Health: {m5['health_score_trend']}")

        st.progress(
            min(max(int(round(m5["health_score"])), 0), 100),
            text=f"Equipment Health Score: {m5['health_score']:.2f}% ({m5['health_status']}) — Formula: 100 × (1 - {m1['failure_probability']:.4f})",
        )

        # Member 2 Sensor Check Table
        st.markdown("##### Member 2 – Sensor Operating Range Verification (IQR Rules)")
        checks_df = pd.DataFrame(m2["sensor_checks"])
        checks_df["Status"] = checks_df["is_abnormal"].map(
            lambda x: "🔴 ABNORMAL" if x else "🟢 Normal"
        )
        checks_df = checks_df.rename(
            columns={
                "sensor": "Sensor Name",
                "value": "Entered Value",
                "normal_min": "Normal Min (IQR)",
                "normal_max": "Normal Max (IQR)",
            }
        )[["Sensor Name", "Entered Value", "Normal Min (IQR)", "Normal Max (IQR)", "Status"]]
        st.dataframe(checks_df, use_container_width=True, hide_index=True)

        # Member 3 Maintenance Recommendations
        st.markdown("##### Member 3 – Failure Category & Maintenance Recommendation")
        st.info(
            f"**Suspected Issue:** `{m3['primary_issue']}`  |  "
            f"**Action Type:** `{m3['action_type']}`  |  "
            f"**Estimated Downtime:** `{m3['estimated_downtime']}`\n\n"
            + "\n".join(f"- {act}" for act in m3["recommended_actions"])
        )

        # Member 4 Alert Notification
        st.markdown("##### Member 4 – Automated Alert Status")
        if m4:
            st.warning(
                f"🔔 **Active Alert (`{m4['status']}`)**: {m4['alert_description']}"
            )
        else:
            st.success("🔕 No active alert generated — equipment is operating within healthy parameters.")

    # Session Submission History Table
    if st.session_state.get("manual_submission_log"):
        st.divider()
        st.markdown("#### 📋 Recent Manual Sensor Predictions in This Session")
        log_rows = []
        for item in st.session_state["manual_submission_log"]:
            log_rows.append(
                {
                    "Timestamp": item["timestamp"],
                    "Equipment ID": item["equipment_id"],
                    "Type": item["equipment_type"],
                    "Prediction (M1)": item["member1_prediction"]["prediction_label"],
                    "Failure Prob (M1)": f"{item['member1_prediction']['failure_probability']*100:.2f}%",
                    "Health Score (M5)": item["member5_health_and_trend"]["health_score"],
                    "Health Status (M5)": item["member5_health_and_trend"]["health_status"],
                    "Risk Level (M2)": item["member2_anomaly"]["risk_level"],
                    "Primary Issue (M3)": item["member3_maintenance"]["primary_issue"],
                    "Saved to DB": "Yes" if item["saved_to_db"] else "No",
                }
            )
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)


def render_fleet_overview_tab(payload: dict) -> None:
    st.subheader("📊 Fleet Health, Risk & Database Overview (2,000 Equipment Units)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Equipment Health Score Classification (Member 5 Part C)")
        h_df = pd.DataFrame(
            {
                "Health Status": ["Good (80-100)", "Moderate (50-79)", "Poor (<50)"],
                "Equipment Count": [
                    payload["health_dist"].get("Good", 0),
                    payload["health_dist"].get("Moderate", 0),
                    payload["health_dist"].get("Poor", 0),
                ],
            }
        ).set_index("Health Status")
        st.bar_chart(h_df, color="#10b981")

    with col2:
        st.markdown("##### Equipment Risk Classification (Member 2)")
        r_df = pd.DataFrame(
            {
                "Risk Level": ["Healthy", "Warning", "Critical"],
                "Equipment Count": [
                    payload["risk_dist"].get("Healthy", 0),
                    payload["risk_dist"].get("Warning", 0),
                    payload["risk_dist"].get("Critical", 0),
                ],
            }
        ).set_index("Risk Level")
        st.bar_chart(r_df, color="#f59e0b")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("##### Identified Failure Categories (Member 3 Rules)")
        issue_df = pd.DataFrame(
            list(payload["issue_dist"].items()),
            columns=["Failure Category", "Count"],
        ).set_index("Failure Category")
        st.bar_chart(issue_df, color="#38bdf8")

    with col4:
        st.markdown("##### Shared SQL Database Table Records (Member 5 Part A/B)")
        db_df = pd.DataFrame(
            list(payload["table_counts"].items()),
            columns=["SQL Table", "Row Count"],
        ).set_index("SQL Table")
        st.bar_chart(db_df, color="#6366f1")


def render_equipment_explorer_tab(payload: dict) -> None:
    st.subheader("🔎 Equipment Health, Risk & Maintenance Explorer")

    eq_df = pd.DataFrame(payload["equipment_rows"])
    f1, f2, f3 = st.columns(3)
    with f1:
        search_q = st.text_input("Search by Equipment ID (e.g., OBS-000001)", "")
    with f2:
        health_filter = st.selectbox("Filter by Health Status", ["All", "Poor", "Moderate", "Good"])
    with f3:
        risk_filter = st.selectbox("Filter by Risk Level", ["All", "Critical", "Warning", "Healthy"])

    if search_q.strip():
        eq_df = eq_df[eq_df["equipment_id"].str.contains(search_q.strip(), case=False, na=False)]
    if health_filter != "All":
        eq_df = eq_df[eq_df["health_status"] == health_filter]
    if risk_filter != "All":
        eq_df = eq_df[eq_df["risk_level"] == risk_filter]

    st.dataframe(eq_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### 🗂️ Inspect Complete SQL Database History for Any Equipment ID")
    selected_eq = st.text_input("Enter Equipment ID to Query SQL History", value="OBS-000001")
    if selected_eq.strip():
        hist = retrieve_equipment_history(selected_eq.strip())
        t_res = analyse_trends(selected_eq.strip())
        hc1, hc2, hc3, hc4 = st.columns(4)
        hc1.metric("Overall Trend", t_res["overall_trend"])
        hc2.metric("Health Score Trend", t_res["health_score_trend"], f"Δ: {t_res['health_score_delta']:+.2f}")
        hc3.metric("Failure Prob Trend", t_res["failure_probability_trend"])
        hc4.metric("Sensor Trend", t_res["sensor_trend"])

        with st.expander("View Raw SQL Records (sensor_readings, predictions, health_scores, maintenance, alerts)", expanded=False):
            st.write("**Predictions Table:**", pd.DataFrame(hist["predictions"]))
            st.write("**Health Scores Table:**", pd.DataFrame(hist["health_scores"]))
            st.write("**Sensor Readings Table:**", pd.DataFrame(hist["sensor_readings"]))
            st.write("**Maintenance Records Table:**", pd.DataFrame(hist["maintenance"]))
            st.write("**Alerts Table:**", pd.DataFrame(hist["alerts"]))


def render_trends_And_alerts_tab(payload: dict) -> None:
    st.subheader("📈 Member 5 Historical Trend Analysis & Member 4 Alert Management")

    st.markdown("##### 7-Day Multi-Timestamp Historical Trend Analysis (`src/trend_analysis.py`)")
    trends_df = pd.DataFrame(payload["tracked_trends"])
    st.dataframe(trends_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("##### 🔔 Active & Historical Equipment Alerts (`alerts` SQL Table)")
    alerts_df = pd.DataFrame(payload["alerts"])
    if not alerts_df.empty:
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            a_risk = st.selectbox("Filter Alert Severity", ["All", "Critical", "Warning"])
        with a_col2:
            a_stat = st.selectbox("Filter Alert Status", ["All", "New", "Acknowledged", "Resolved"])

        filtered_alerts = alerts_df.copy()
        if a_risk != "All":
            filtered_alerts = filtered_alerts[filtered_alerts["risk_level"] == a_risk]
        if a_stat != "All":
            filtered_alerts = filtered_alerts[filtered_alerts["status"] == a_stat]

        st.dataframe(
            filtered_alerts[
                ["alert_id", "equipment_id", "ts", "risk_level", "failure_probability", "condition", "status", "description"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("###### Update Alert Status in SQL Database")
        u_col1, u_col2, u_col3 = st.columns([2, 1, 1])
        with u_col1:
            target_alert_id = st.selectbox(
                "Select Alert ID",
                options=filtered_alerts["alert_id"].tolist() if not filtered_alerts.empty else alerts_df["alert_id"].tolist(),
            )
        with u_col2:
            new_status = st.selectbox("New Status", ["Acknowledged", "Resolved"])
        with u_col3:
            st.write("")
            st.write("")
            if st.button("Update Alert Status in DB", use_container_width=True):
                try:
                    update_alert_status_in_db(target_alert_id, new_status)
                    st.success(f"Updated Alert `{target_alert_id}` to `{new_status}` in SQL Database!")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))


def main() -> None:
    payload = fetch_dashboard_payload()
    render_header_and_kpis(payload)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🧪 1. Manual Sensor Input & Failure Prediction",
            "📊 2. Fleet Overview & Charts",
            "🔎 3. Equipment Explorer & SQL History",
            "📈 4. Historical Trends & Alert Management",
        ]
    )

    with tab1:
        render_manual_prediction_tab()
    with tab2:
        render_fleet_overview_tab(payload)
    with tab3:
        render_equipment_explorer_tab(payload)
    with tab4:
        render_trends_And_alerts_tab(payload)


if __name__ == "__main__":
    main()
