"""
Advanced Task T44 – Simulated Real-Time Sensor-Monitoring Pipeline
Expected Submission: Architecture diagram and prototype
(reports/t44_pipeline_architecture_and_prototype.md,
 reports/t44_pipeline_architecture_diagram.png,
 reports/t44_simulated_stream_log.csv)

Implements a 6-stage simulated industrial IoT sensor-monitoring streaming
prototype that generates sequential sensor telemetry ticks, validates and
engineers features in real time, runs dual ML inference (Binary XGBoost +
Multiclass Failure-Type + TreeSHAP), computes 0-100 health scores and IQR
anomalies, and logs alert/maintenance triggers.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

from src.prediction_service import create_prediction_service
from src.analysis_service import run_post_prediction_analysis
from src.failure_type_model import REPORTS_DIR


def draw_pipeline_architecture_diagram() -> str:
    """Draw a high-resolution 6-stage architecture diagram and save to reports/."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    ax.text(
        6.25,
        4.82,
        "T44: Real-Time Simulated Industrial Sensor-Monitoring Pipeline Architecture",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
    )

    stages = [
        (0.3, 2.2, 1.75, 1.8, "#1e293b", "1. IoT Telemetry\nSimulator", "• Air & Process Temp [K]\n• Speed [rpm]\n• Torque [Nm]\n• Tool Wear [min]"),
        (2.35, 2.2, 1.75, 1.8, "#0284c7", "2. Stream Buffer\n& Validator", "• Schema & Range Gate\n• Finite Value Check\n• Equipment Registry\n  Lookup (L/M/H)"),
        (4.4, 2.2, 1.75, 1.8, "#4f46e5", "3. Real-Time\nFeature Engine", "• ΔT = T_proc - T_air\n• One-Hot Type (H/L/M)\n• Bracket Sanitizer\n  for XGBoost DMatrix"),
        (6.45, 2.2, 1.85, 1.8, "#7c3aed", "4. Dual ML &\nSHAP Inference", "• Binary XGBoost (P_fail)\n• 6-Class Failure Mode\n• Exact TreeSHAP\n  Top Feature Drivers"),
        (8.6, 2.2, 1.75, 1.8, "#0d9488", "5. Health, IQR &\nRisk Engine", "• Health = 100*(1 - P)\n• 5-Sensor IQR Gate\n• Risk: Healthy /\n  Warning / Critical"),
        (10.65, 2.2, 1.6, 1.8, "#e11d48", "6. Persistence &\nAlert Dispatch", "• PostgreSQL / SQLite\n• Deduplicated Alerts\n• AI4I Maintenance\n• Live React UI"),
    ]

    for x, y, w, h, color, title, body in stages:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            linewidth=1.5,
            edgecolor=color,
            facecolor="#ffffff",
        )
        ax.add_patch(rect)
        header = patches.FancyBboxPatch(
            (x, y + h - 0.52),
            w,
            0.52,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=0,
            facecolor=color,
        )
        ax.add_patch(header)
        ax.text(x + w / 2, y + h - 0.26, title, ha="center", va="center", fontsize=8.8, fontweight="bold", color="#ffffff")
        ax.text(x + 0.1, y + h - 0.68, body, ha="left", va="top", fontsize=7.5, color="#334155", linespacing=1.38)

    # Connect stages with arrows
    arrow_xs = [(2.12, 2.30), (4.17, 4.35), (6.22, 6.40), (8.37, 8.55), (10.42, 10.60)]
    for x1, x2 in arrow_xs:
        ax.annotate(
            "",
            xy=(x2, 3.1),
            xytext=(x1, 3.1),
            arrowprops=dict(arrowstyle="->", color="#475569", lw=2.0),
        )

    # Bottom feedback & storage bar
    storage_box = patches.FancyBboxPatch(
        (1.5, 0.45),
        9.5,
        1.1,
        boxstyle="round,pad=0.08,rounding_size=0.15",
        linewidth=1.2,
        edgecolor="#94a3b8",
        facecolor="#f8fafc",
    )
    ax.add_patch(storage_box)
    ax.text(
        6.25,
        1.15,
        "Shared SQL Telemetry Store (PostgreSQL / SQLite) & Historical Trend Analyzer",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        6.25,
        0.75,
        "Tables: equipment · sensor_readings · predictions · anomaly_results · health_scores · maintenance_records · alerts  |  Trajectories: Improving / Stable / Deteriorating",
        ha="center",
        va="center",
        fontsize=8,
        color="#475569",
    )

    plt.tight_layout()
    diagram_path = REPORTS_DIR / "t44_pipeline_architecture_diagram.png"
    fig.savefig(diagram_path, dpi=160)
    plt.close(fig)
    return str(diagram_path)


def run_simulated_sensor_pipeline(
    equipment_id: str = "SIM-CNC-01",
    equipment_type: str = "L",
    num_steps: int = 20,
    save_to_db: bool = False,
) -> Dict[str, Any]:
    """Run a 20-step simulated sensor stream from Normal -> Degradation -> Critical Failure -> Recovery."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    diagram_path = draw_pipeline_architecture_diagram()
    predict_fn = create_prediction_service()

    base_time = datetime(2026, 11, 19, 12, 0, 0, tzinfo=timezone.utc)
    rng = np.random.default_rng(44)
    stream_rows: List[Dict[str, Any]] = []

    for step in range(1, num_steps + 1):
        ts = (base_time + timedelta(minutes=15 * step)).isoformat(timespec="seconds")

        if step <= 8:
            # Phase 1: Healthy steady-state machining
            phase = "Phase 1: Normal Operation"
            air_t = round(298.2 + step * 0.1 + float(rng.normal(0, 0.08)), 2)
            proc_t = round(air_t + 10.2 + float(rng.normal(0, 0.08)), 2)
            speed = round(1540 - step * 8 + float(rng.normal(0, 6)), 0)
            torque = round(39.5 + step * 0.7 + float(rng.normal(0, 0.4)), 2)
            wear = round(20 + step * 14, 0)
        elif step <= 14:
            # Phase 2: Accelerated wear & rising torque / narrowing delta-T
            phase = "Phase 2: Early Degradation"
            air_t = round(299.5 + (step - 8) * 0.25, 2)
            proc_t = round(air_t + 9.2 - (step - 8) * 0.12, 2)
            speed = round(1440 - (step - 8) * 18, 0)
            torque = round(48.0 + (step - 8) * 2.2, 2)
            wear = round(135 + (step - 8) * 12, 0)
        elif step <= 18:
            # Phase 3: Critical Overstrain & Heat Dissipation Failure zone
            phase = "Phase 3: Critical Failure Stress"
            air_t = round(301.8 + (step - 14) * 0.35, 2)
            proc_t = round(air_t + 7.9 - (step - 14) * 0.1, 2)
            speed = round(1330 - (step - 14) * 15, 0)
            torque = round(63.0 + (step - 14) * 2.5, 2)
            wear = round(210 + (step - 14) * 8, 0)
        else:
            # Phase 4: Post-maintenance tool replacement & cooldown
            phase = "Phase 4: Post-Maintenance Reset"
            air_t = 298.0
            proc_t = 308.5
            speed = 1560.0
            torque = 38.0
            wear = float((step - 19) * 5)

        readings = {
            "equipment_id": equipment_id,
            "Type": equipment_type,
            "Air temperature [K]": air_t,
            "Process temperature [K]": proc_t,
            "Rotational speed [rpm]": speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear,
        }

        pred = predict_fn(readings)
        analysis = run_post_prediction_analysis(
            readings=readings,
            prediction_result=pred,
            timestamp=ts,
            save_to_db=save_to_db,
        )

        top_shap = (
            f"{analysis['shap_drivers'][0]['feature']} ({analysis['shap_drivers'][0]['shap_contribution']:+.2f})"
            if analysis.get("shap_drivers")
            else "N/A"
        )

        stream_rows.append(
            {
                "Step": step,
                "Timestamp": ts,
                "Equipment ID": equipment_id,
                "Operating Phase": phase,
                "Air Temp [K]": air_t,
                "Process Temp [K]": proc_t,
                "Speed [rpm]": int(speed),
                "Torque [Nm]": torque,
                "Tool Wear [min]": int(wear),
                "Prediction": pred["prediction_label"],
                "Failure Probability": round(float(pred["failure_probability"]), 4),
                "Health Score": analysis["health_score"],
                "Health Status": analysis["health_status"],
                "Risk Level": analysis["risk_level"],
                "Specific Failure Mode (T38)": analysis.get("specific_failure_type", analysis["primary_issue"]),
                "Top SHAP Driver (T41)": top_shap,
                "Recommended Action": analysis["action_type"],
            }
        )

    df_stream = pd.DataFrame(stream_rows)
    csv_path = REPORTS_DIR / "t44_simulated_stream_log.csv"
    df_stream.to_csv(csv_path, index=False)

    md_table = "\n".join(
        f"| `{r['Step']}` | {r['Operating Phase']} | `{r['Air Temp [K]']}` | `{r['Process Temp [K]']}` | `{r['Speed [rpm]']}` | `{r['Torque [Nm]']}` | `{r['Tool Wear [min]']}` | **{r['Prediction']}** (`{r['Failure Probability']*100:.1f}%`) | `{r['Health Score']:.1f}` ({r['Health Status']}) | **{r['Risk Level']}** | {r['Specific Failure Mode (T38)']} |"
        for r in stream_rows
    )

    md_content = f"""# Advanced Task T44: Simulated Sensor-Monitoring Pipeline Architecture & Prototype

## 1. Architecture Diagram (Mermaid & Visual PNG)

![Pipeline Architecture Diagram](t44_pipeline_architecture_diagram.png)

```mermaid
flowchart LR
    A["1. IoT Sensor Simulator<br/>(5 Telemetry Streams @ 15m ticks)"] --> B["2. Ingestion & Validator<br/>(Schema, Finite Bounds & Registry)"]
    B --> C["3. Real-Time Feature Engine<br/>(ΔT = T_proc - T_air & Type One-Hot)"]
    C --> D["4. Dual ML & SHAP Inference<br/>(Binary XGBoost + 6-Class Mode + TreeSHAP)"]
    D --> E["5. Health & IQR Risk Gate<br/>(0-100 Health Score & Healthy/Warning/Critical)"]
    E --> F["6. SQL Persistence & Alert Dispatch<br/>(PostgreSQL/SQLite + Multi-Channel Alerts)"]
```

---

## 2. Simulated 20-Tick Progressive Degradation Stream (`SIM-CNC-01`)

The prototype (`src/sensor_monitoring_pipeline.py`) simulates a CNC spindle progressing through four realistic physical regimes:
1. **Steps 1–8 (`Phase 1: Normal Operation`)**: Healthy baseline operation (`Health Score = 99–100`, `Risk = Healthy`).
2. **Steps 9–14 (`Phase 2: Early Degradation`)**: Rising tool wear and torque trigger early `Warning` risk classification before failure occurs.
3. **Steps 15–18 (`Phase 3: Critical Failure Stress`)**: High tool wear ($>215\\text{{ min}}$), high torque ($>65\\text{{ Nm}}$), and narrow $\\Delta T$ trigger `Failure` predictions (`Probability > 85%`, `Health Score < 15`, `Risk = Critical`) with specific failure mode attribution and emergency maintenance dispatch.
4. **Steps 19–20 (`Phase 4: Post-Maintenance Reset`)**: Tool replacement restores health score to `99.9 / 100` (`Healthy`).

| Step | Operating Phase | Air [K] | Proc [K] | Speed [rpm] | Torque [Nm] | Wear [min] | Prediction (Prob) | Health Score | Risk Level | Specific Mode (T38) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
{md_table}
"""
    md_path = REPORTS_DIR / "t44_pipeline_architecture_and_prototype.md"
    md_path.write_text(md_content, encoding="utf-8")

    return {
        "diagram_path": diagram_path,
        "csv_path": str(csv_path),
        "md_path": str(md_path),
        "stream_rows": stream_rows,
    }


if __name__ == "__main__":
    res = run_simulated_sensor_pipeline()
    print("T44 Pipeline Architecture & Prototype saved to:", res["md_path"])
