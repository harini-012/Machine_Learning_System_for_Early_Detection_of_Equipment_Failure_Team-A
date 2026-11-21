"""
Generate Member 5's 3-Page PDF & PPTX Presentation:
  - Slide 1: What We Did at the Final Stage (Milestone 4 Team Integration + Member 5's Key Tasks & Deliverables)
  - Slide 2 (Visuals 1): Held-Out Test Metrics Bar Chart + 2x2 Confusion Matrix Heatmap + Threshold Curve (T42) + TreeSHAP Drivers (T41)
  - Slide 3 (Visuals 2): Render Cloud Architecture Flowchart + Live Demo Trajectory Chart + Fleet Risk/Test Donut + Cross-Plant Generalization Chart
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import seaborn as sns
from pptx import Presentation
from pptx.util import Inches

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_DIR / "reports"


def _add_slide_banner(fig, slide_label: str, title: str, badge_text: str, badge_color: str = "#2563eb"):
    banner_ax = fig.add_axes([0.03, 0.885, 0.94, 0.088])
    banner_ax.set_xlim(0, 10)
    banner_ax.set_ylim(0, 1)
    banner_ax.axis("off")

    bg = patches.FancyBboxPatch(
        (0, 0),
        10,
        1,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor="#0f172a",
        edgecolor="none",
    )
    banner_ax.add_patch(bg)
    banner_ax.text(0.22, 0.50, f"{slide_label}  ·  {title}", fontsize=14.8, fontweight="bold", color="#ffffff", va="center")

    badge = patches.FancyBboxPatch(
        (8.15, 0.18),
        1.68,
        0.64,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor=badge_color,
        edgecolor="none",
    )
    banner_ax.add_patch(badge)
    banner_ax.text(8.99, 0.50, badge_text, fontsize=10.8, fontweight="bold", color="#ffffff", ha="center", va="center")


def make_slide_1_overview():
    """Slide 1: Final Stage Project Summary & Member 5's Key Tasks."""
    fig = plt.figure(figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor("#f8fafc")
    _add_slide_banner(
        fig,
        "SLIDE 1 (SUMMARY)",
        "Final Stage Integration & Member 5 Key Tasks (Testing, Deploy & Docs)",
        "MEMBER 5 ROLE",
        badge_color="#059669",
    )

    ax = fig.add_axes([0.03, 0.04, 0.94, 0.82])
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    # Top Strip: How the 5 Members Connected in Milestone 4
    strip_card = patches.FancyBboxPatch(
        (0.05, 5.95),
        15.9,
        1.52,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        linewidth=1.3,
    )
    ax.add_patch(strip_card)
    ax.text(0.35, 7.15, "Final Stage Team Workflow (Milestone 4 End-to-End Sequential Handoff)", fontsize=12.0, fontweight="bold", color="#0f172a")

    m_steps = [
        (0.35, "M1 · Flask & Model", "best_model.joblib\n& /predict form"),
        (3.50, "M2 · UI Dashboard", "Responsive layout\n& React 18 Telemetry UI"),
        (6.65, "M3 · Equipment & DB", "PostgreSQL / SQLite CRUD\n& saved history charts"),
        (9.80, "M4 · Alerts & Reports", "Health score, IQR risk,\nmaintenance & 3 CSVs"),
        (12.95, "M5 · Test & Deploy (Me)", "27/27 tests, held-out eval,\nRender deploy & README"),
    ]
    for idx, (mx, mtitle, msub) in enumerate(m_steps):
        is_m5 = idx == 4
        mcol = "#059669" if is_m5 else "#2563eb"
        mbg = "#ecfdf5" if is_m5 else "#eff6ff"
        mbox = patches.FancyBboxPatch((mx, 6.10), 2.75, 0.85, boxstyle="round,pad=0.02,rounding_size=0.1", facecolor=mbg, edgecolor=mcol, linewidth=1.7 if is_m5 else 1.1)
        ax.add_patch(mbox)
        ax.text(mx + 1.375, 6.72, mtitle, ha="center", va="center", fontsize=10.0, fontweight="bold", color=mcol)
        ax.text(mx + 1.375, 6.35, msub, ha="center", va="center", fontsize=8.6, color="#334155", linespacing=1.25)
        if idx < 4:
            ax.annotate("", xy=(mx + 3.08, 6.52), xytext=(mx + 2.80, 6.52), arrowprops=dict(arrowstyle="->", color="#64748b", lw=2.0))

    # Bottom 2 Columns: Member 5's 6 Key Deliverables
    left_card = patches.FancyBboxPatch((0.05, 0.08), 7.75, 5.65, boxstyle="round,pad=0.02,rounding_size=0.14", facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.3)
    right_card = patches.FancyBboxPatch((8.20, 0.08), 7.75, 5.65, boxstyle="round,pad=0.02,rounding_size=0.14", facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.3)
    ax.add_patch(left_card)
    ax.add_patch(right_card)

    ax.text(0.35, 5.38, "Member 5 Core Tasks: Testing & Model Verification", fontsize=12.5, fontweight="bold", color="#1e40af")
    ax.text(8.50, 5.38, "Member 5 Core Tasks: Cloud Deployment, Demo & Advanced ML", fontsize=12.5, fontweight="bold", color="#047857")

    left_tasks = [
        (
            "1. End-to-End Automated Testing (test_milestone4_workflow.py)",
            "#2563eb",
            "#eff6ff",
            "• Verified all pages load (/, /dashboard, /predict, /equipment, /monitoring/*).\n"
            "• Tested Equipment Add/Edit/Delete & invalid sensor input validation (HTTP 400).\n"
            "• Verified history persistence across reopen, alert status updates, & 3 CSV exports.",
        ),
        (
            "2. Held-Out Test Set Model Verification (test_model_performance.py)",
            "#059669",
            "#ecfdf5",
            "• Evaluated saved XGBClassifier on 2,000 held-out test records (68 failures):\n"
            "  Accuracy: 99.05%  |  Precision: 96.23% (2 FP)  |  Recall: 75.00%  |  F1: 84.30%\n"
            "• Confirmed single-reading Flask service matches batch predictions identically.",
        ),
        (
            "3. Decision Threshold & Precision-Recall Analysis (Task T42)",
            "#7c3aed",
            "#f5f3ff",
            "• Examined 11 thresholds t in [0.10, 0.90] (PR-AUC = 88.24%, ROC-AUC = 97.50%).\n"
            "• Proved 3-tier policy: t=0.30 (Warning), t=0.50 (Peak F1), t=0.70 (Critical 97.8% Prec).",
        ),
    ]
    y = 4.98
    for ttitle, tcol, tbg, tbody in left_tasks:
        tbox = patches.FancyBboxPatch((0.35, y - 1.38), 7.15, 1.46, boxstyle="round,pad=0.02,rounding_size=0.1", facecolor=tbg, edgecolor=tcol, linewidth=1.2)
        ax.add_patch(tbox)
        ax.text(0.58, y - 0.18, ttitle, fontsize=10.4, fontweight="bold", color=tcol)
        ax.text(0.58, y - 0.46, tbody, fontsize=8.9, color="#1e293b", va="top", linespacing=1.38)
        y -= 1.62

    right_tasks = [
        (
            "4. Render Cloud Deployment (Procfile, render.yaml, requirements.txt)",
            "#0284c7",
            "#f0f9ff",
            "• Deployed live at: https://machine-learning-system-for-early.onrender.com\n"
            "• Configured Gunicorn server, packaged trained models & selected features, and\n"
            "  connected dual PostgreSQL (DATABASE_URL) + SQLite (12,220 records, 7 tables).",
        ),
        (
            "5. Live Deployed URL Demo & Persistence Verification",
            "#d97706",
            "#fffbeb",
            "• Registered EQ-RENDER-01 on live site -> Submitted Normal (No Failure, 99.9 Health)\n"
            "  & High-Risk (Failure 99.65%, Critical Alert + Maintenance Plan) -> Reopened history\n"
            "  page to confirm saved readings & charts persist -> Downloaded all 3 CSV reports.",
        ),
        (
            "6. Documentation (README.md) & Advanced ML Suite (T38–T45)",
            "#e11d48",
            "#fff1f2",
            "• Added live URL, setup guide, test results, and screenshots to README.md.\n"
            "• Built 6-Class Failure-Type Model (T38/T39: 97.95% Acc), Isolation Forest (T40),\n"
            "  Exact TreeSHAP (T41), Generalization (T43), Stream Pipeline (T44), & Alerts (T45).",
        ),
    ]
    y = 4.98
    for ttitle, tcol, tbg, tbody in right_tasks:
        tbox = patches.FancyBboxPatch((8.50, y - 1.38), 7.15, 1.46, boxstyle="round,pad=0.02,rounding_size=0.1", facecolor=tbg, edgecolor=tcol, linewidth=1.2)
        ax.add_patch(tbox)
        ax.text(8.73, y - 0.18, ttitle, fontsize=10.4, fontweight="bold", color=tcol)
        ax.text(8.73, y - 0.46, tbody, fontsize=8.9, color="#1e293b", va="top", linespacing=1.38)
        y -= 1.62

    return fig


def make_slide_2_model_visuals():
    """Slide 2 (Visuals 1): Held-Out Metrics Bar Chart + Confusion Matrix Heatmap + Threshold Curve (T42) + TreeSHAP (T41)."""
    fig = plt.figure(figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor("#f8fafc")
    _add_slide_banner(
        fig,
        "SLIDE 2 (VISUALS 1)",
        "Held-Out Model Performance, Confusion Matrix, Threshold Curve & TreeSHAP",
        "99.05% ACCURACY",
        badge_color="#2563eb",
    )

    ax1 = fig.add_axes([0.06, 0.50, 0.40, 0.32])
    ax2 = fig.add_axes([0.55, 0.50, 0.39, 0.32])
    ax3 = fig.add_axes([0.06, 0.08, 0.40, 0.33])
    ax4 = fig.add_axes([0.56, 0.08, 0.38, 0.33])

    # 1. Top-Left: Held-Out Test Set Metrics Bar Chart
    metrics = ["Recall", "F1-Score", "PR-AUC", "Precision", "ROC-AUC", "Accuracy"]
    values = [75.00, 84.30, 88.24, 96.23, 97.50, 99.05]
    colors = ["#f59e0b", "#0d9488", "#6366f1", "#10b981", "#3b82f6", "#1d4ed8"]

    bars = ax1.barh(metrics, values, color=colors, height=0.58)
    ax1.set_xlim(0, 112)
    ax1.set_title("1. Held-Out Test Set Metrics (2,000 Units · XGBoost)", fontsize=12, fontweight="bold", color="#0f172a", pad=8)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    for bar, v in zip(bars, values):
        ax1.text(v + 1.2, bar.get_y() + bar.get_height() / 2, f"{v:.2f}%", va="center", fontsize=9.8, fontweight="bold", color="#0f172a")

    # 2. Top-Right: 2x2 Confusion Matrix Heatmap
    annot = np.array(
        [
            ["TN = 1,930\n(99.9% Normal)", "FP = 2\n(0.1% False Alarm)"],
            ["FN = 17\n(Missed @ t=0.50)", "TP = 51\n(75.0% Caught)"],
        ]
    )
    sns.heatmap(
        np.array([[0.95, 0.08], [0.25, 0.78]]),
        annot=annot,
        fmt="",
        cmap="YlGnBu",
        cbar=False,
        xticklabels=["Predicted: No Failure", "Predicted: Failure"],
        yticklabels=["Actual: No Failure", "Actual: Failure"],
        annot_kws={"size": 10.5, "weight": "bold"},
        linewidths=2.5,
        linecolor="#ffffff",
        ax=ax2,
    )
    ax2.set_title("2. Held-Out Confusion Matrix (68 True Failures)", fontsize=12, fontweight="bold", color="#0f172a", pad=8)

    # 3. Bottom-Left: T42 Decision Threshold Curve
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    precision = [77.14, 85.25, 88.14, 92.73, 96.23, 96.00, 97.78, 97.62, 100.0]
    recall = [79.41, 76.47, 76.47, 75.00, 75.00, 70.59, 64.71, 60.29, 48.53]
    f1 = [78.26, 80.62, 81.89, 82.93, 84.30, 81.36, 77.88, 74.55, 65.35]

    ax3.plot(thresholds, precision, "o-", color="#2563eb", linewidth=2.3, label="Precision (%)")
    ax3.plot(thresholds, recall, "s-", color="#e11d48", linewidth=2.3, label="Recall (%)")
    ax3.plot(thresholds, f1, "^-", color="#059669", linewidth=2.6, label="F1-Score (%)")
    ax3.axvline(0.30, color="#f59e0b", linestyle="--", linewidth=1.4, label="Warning (t=0.30)")
    ax3.axvline(0.50, color="#059669", linestyle="--", linewidth=1.4, label="Peak F1 (t=0.50)")
    ax3.axvline(0.70, color="#e11d48", linestyle="--", linewidth=1.4, label="Critical (t=0.70)")
    ax3.set_ylim(44, 104)
    ax3.set_xlabel("Decision Threshold (t)", fontsize=9.5, fontweight="bold")
    ax3.set_ylabel("Score (%)", fontsize=9.5, fontweight="bold")
    ax3.set_title("3. Decision Threshold Precision-Recall Curve (T42)", fontsize=12, fontweight="bold", color="#0f172a", pad=8)
    ax3.legend(loc="lower left", fontsize=8, ncol=2)
    ax3.grid(linestyle="--", alpha=0.4)
    for spine in ["top", "right"]:
        ax3.spines[spine].set_visible(False)

    # 4. Bottom-Right: T41 Exact TreeSHAP Feature Importance
    features = [
        "Type_H / M / L",
        "Process Temp [K]",
        "Air Temp [K]",
        "Temp Diff ΔT [K]",
        "Tool Wear [min]",
        "Torque [Nm]",
        "Speed [rpm]",
    ]
    shap_shares = [4.5, 6.2, 7.8, 12.4, 18.9, 24.1, 26.1]
    shap_colors = ["#94a3b8", "#64748b", "#38bdf8", "#0d9488", "#f59e0b", "#6366f1", "#4f46e5"]

    bars4 = ax4.barh(features, shap_shares, color=shap_colors, height=0.58)
    ax4.set_xlim(0, 31)
    ax4.set_xlabel("Share of Total |SHAP| Impact (%)", fontsize=9.5, fontweight="bold")
    ax4.set_title("4. Exact TreeSHAP Top Feature Drivers (T41)", fontsize=12, fontweight="bold", color="#0f172a", pad=8)
    ax4.grid(axis="x", linestyle="--", alpha=0.4)
    for spine in ["top", "right"]:
        ax4.spines[spine].set_visible(False)
    for b, val in zip(bars4, shap_shares):
        ax4.text(val + 0.5, b.get_y() + b.get_height() / 2, f"{val:.1f}%", va="center", fontsize=9.2, fontweight="bold", color="#0f172a")

    return fig


def make_slide_3_deployment_visuals():
    """Slide 3 (Visuals 2): Render Cloud Architecture Flowchart + Live Demo Trajectory + Test/Risk Donut + Generalization Chart."""
    fig = plt.figure(figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor("#f8fafc")
    _add_slide_banner(
        fig,
        "SLIDE 3 (VISUALS 2)",
        "Cloud Architecture Map, Live Demo Trajectory & Cross-Plant Generalization",
        "LIVE ON RENDER",
        badge_color="#059669",
    )

    # 1. Top Panel: Visual Cloud Architecture Flowchart
    flow_ax = fig.add_axes([0.03, 0.56, 0.94, 0.28])
    flow_ax.set_xlim(0, 15)
    flow_ax.set_ylim(0, 3)
    flow_ax.axis("off")

    flow_ax.text(
        0.15,
        2.72,
        "1. Production Cloud Deployment Map  (https://machine-learning-system-for-early.onrender.com)",
        fontsize=12.5,
        fontweight="bold",
        color="#0f172a",
    )

    nodes = [
        (0.15, 0.35, 2.55, 2.0, "#1e293b", "React 18 Web UI", "Prediction Studio\nEquipment Fleet\nTelemetry Cards"),
        (3.20, 0.35, 2.55, 2.0, "#2563eb", "Render Cloud Host", "Gunicorn WSGI\nFlask App (app.py)\n/health Endpoint"),
        (6.25, 0.35, 2.55, 2.0, "#7c3aed", "XGBoost + SHAP", "best_model.joblib\n6-Class Mode Model\nExact TreeSHAP"),
        (9.30, 0.35, 2.55, 2.0, "#0d9488", "Shared SQL DB", "PostgreSQL / SQLite\n7 Normalized Tables\n12,220 Records"),
        (12.35, 0.35, 2.50, 2.0, "#e11d48", "Alerts & CSVs", "Health Score 0-100\nAlert Lifecycle\n3 CSV Exports"),
    ]
    for nx, ny, nw, nh, ncol, ntitle, nsub in nodes:
        box = patches.FancyBboxPatch((nx, ny), nw, nh, boxstyle="round,pad=0.03,rounding_size=0.18", facecolor="#ffffff", edgecolor=ncol, linewidth=2.0)
        flow_ax.add_patch(box)
        top = patches.FancyBboxPatch((nx, ny + nh - 0.58), nw, 0.58, boxstyle="round,pad=0.01,rounding_size=0.14", facecolor=ncol, edgecolor="none")
        flow_ax.add_patch(top)
        flow_ax.text(nx + nw / 2, ny + nh - 0.29, ntitle, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#ffffff")
        flow_ax.text(nx + nw / 2, ny + 0.68, nsub, ha="center", va="center", fontsize=9.2, color="#334155", linespacing=1.35)

    for ax_x in [2.75, 5.80, 8.85, 11.90]:
        flow_ax.annotate("", xy=(ax_x + 0.40, 1.35), xytext=(ax_x, 1.35), arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.35", color="#475569", lw=2.5))

    # 2. Bottom-Left: Live Demo Trajectory Chart (EQ-RENDER-01)
    ax_traj = fig.add_axes([0.05, 0.09, 0.33, 0.40])
    obs_labels = ["Obs 1\nNormal", "Obs 2\nSteady", "Obs 3\nDrift", "Obs 4\nStress", "Obs 5\nCritical"]
    health_scores = [99.94, 96.40, 62.15, 28.50, 0.35]
    fail_probs = [0.06, 3.60, 37.85, 71.50, 99.65]
    x_idx = np.arange(len(obs_labels))

    ax_traj.plot(x_idx, health_scores, "o-", color="#059669", linewidth=2.8, markersize=7, label="Health Score (0–100)")
    ax_traj.plot(x_idx, fail_probs, "s--", color="#e11d48", linewidth=2.8, markersize=7, label="Failure Prob (%)")
    ax_traj.fill_between(x_idx, health_scores, color="#10b981", alpha=0.08)
    ax_traj.fill_between(x_idx, fail_probs, color="#e11d48", alpha=0.08)

    for i, (hs, fp) in enumerate(zip(health_scores, fail_probs)):
        ax_traj.annotate(f"{hs:.1f}", (x_idx[i], hs), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color="#047857")
        ax_traj.annotate(f"{fp:.1f}%", (x_idx[i], fp), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color="#be123c")

    ax_traj.set_ylim(-14, 116)
    ax_traj.set_xticks(x_idx)
    ax_traj.set_xticklabels(obs_labels, fontsize=8.8, fontweight="bold")
    ax_traj.set_title("2. Live URL Demo Trajectory (EQ-RENDER-01)", fontsize=11.5, fontweight="bold", color="#0f172a", pad=8)
    ax_traj.legend(loc="center left", fontsize=8.2)
    ax_traj.grid(linestyle="--", alpha=0.4)
    for spine in ["top", "right"]:
        ax_traj.spines[spine].set_visible(False)

    # 3. Bottom-Center: Automated Test Suite Donut (27/27)
    ax_donut = fig.add_axes([0.41, 0.13, 0.22, 0.36])
    w1, _ = ax_donut.pie(
        [6, 8, 7, 4, 2],
        colors=["#2563eb", "#7c3aed", "#059669", "#0d9488", "#f59e0b"],
        startangle=120,
        radius=0.95,
        wedgeprops=dict(width=0.42, edgecolor="#f8fafc", linewidth=2.2),
    )
    ax_donut.text(0, 0.06, "27 / 27", ha="center", va="center", fontsize=15, fontweight="bold", color="#059669")
    ax_donut.text(0, -0.15, "100% PASS", ha="center", va="center", fontsize=8.8, fontweight="bold", color="#475569")
    ax_donut.set_title("3. Automated Test Suites", fontsize=11.5, fontweight="bold", color="#0f172a")
    ax_donut.legend(
        w1,
        ["M4 Workflow (6/6)", "Advanced T38-T45 (8/8)", "Alerts Lifecycle (7/7)", "DB & Health (4/4)", "Model Parity (2/2)"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=7.8,
         ncol=2,
        frameon=False,
    )

    # 4. Bottom-Right: Cross-Plant Generalization & Multiclass Bar Chart (T38 / T43)
    ax_gen = fig.add_axes([0.67, 0.09, 0.29, 0.40])
    plants = ["Plant A\nBase", "Plant B\n+3.5K", "Plant C\nLoad", "Plant D\nNoisy", "T38 6-Class\nModel"]
    acc_vals = [99.05, 98.40, 97.60, 98.00, 97.95]
    auc_vals = [97.50, 96.80, 95.90, 96.10, 98.00]
    x = np.arange(len(plants))
    w = 0.34

    b1 = ax_gen.bar(x - w / 2, acc_vals, w, label="Accuracy (%)", color="#2563eb")
    b2 = ax_gen.bar(x + w / 2, auc_vals, w, label="ROC-AUC / Wt-F1 (%)", color="#10b981")
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax_gen.text(bar.get_x() + bar.get_width() / 2, h + 0.25, f"{h:.1f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")

    ax_gen.set_ylim(88, 102)
    ax_gen.set_xticks(x)
    ax_gen.set_xticklabels(plants, fontsize=8.5, fontweight="bold")
    ax_gen.set_title("4. Generalization & Multiclass (T38/T43)", fontsize=11.5, fontweight="bold", color="#0f172a", pad=8)
    ax_gen.legend(loc="lower right", fontsize=8)
    ax_gen.grid(axis="y", linestyle="--", alpha=0.4)
    for spine in ["top", "right"]:
        ax_gen.spines[spine].set_visible(False)

    return fig


def generate_all():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = REPORTS_DIR / "Member5_Milestone4_Presentation.pdf"

    png_paths = [
        REPORTS_DIR / "m5_m4_slide1_testing_and_model_verification.png",
        REPORTS_DIR / "m5_m4_slide2_render_deployment_and_live_demo.png",
        REPORTS_DIR / "m5_m4_slide3_advanced_tasks_and_documentation.png",
    ]
    makers = [make_slide_1_overview, make_slide_2_model_visuals, make_slide_3_deployment_visuals]

    with PdfPages(pdf_path) as pdf:
        for maker, png_path in zip(makers, png_paths):
            fig = maker()
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            fig.savefig(png_path, dpi=160, facecolor=fig.get_facecolor())
            plt.close(fig)

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    for img_path in png_paths:
        slide = prs.slides.add_slide(blank_layout)
        slide.shapes.add_picture(str(img_path), Inches(0), Inches(0), width=Inches(13.333), height=Inches(7.5))
    pptx_path = REPORTS_DIR / "Member5_Milestone4_Demo_Slides.pptx"
    prs.save(str(pptx_path))

    print("Saved 3-Page PDF (1 Overview + 2 Visual Slides):", pdf_path)
    print("Saved 3-Slide PPTX:", pptx_path)


if __name__ == "__main__":
    generate_all()
