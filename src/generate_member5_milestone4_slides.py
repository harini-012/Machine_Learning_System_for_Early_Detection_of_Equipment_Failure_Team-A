"""
Generate a Visual-First, Chart-Rich 3-Page PDF & PPTX Presentation
for Member 5 (Milestone 4) with minimal text and rich data visualizations:
  - Slide 1: Held-Out Test Metric Bar Chart + 2x2 Confusion Matrix Heatmap + 27/27 Test Suite Donut Chart
  - Slide 2: Render Cloud Architecture Flowchart + Live Demo Health/Failure Trajectory Chart + Fleet Risk Donut
  - Slide 3: Decision Threshold Precision-Recall Curve (T42) + Exact TreeSHAP Feature Importance Chart (T41) + Cross-Plant Generalization Chart (T43)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from pptx import Presentation
from pptx.util import Inches

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_DIR / "reports"


def _add_slide_banner(fig, slide_label: str, title: str, badge_text: str, badge_color: str = "#2563eb"):
    banner_ax = fig.add_axes([0.03, 0.88, 0.94, 0.09])
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
    banner_ax.text(0.22, 0.50, f"{slide_label}  ·  {title}", fontsize=14.5, fontweight="bold", color="#ffffff", va="center")

    badge = patches.FancyBboxPatch(
        (8.15, 0.18),
        1.68,
        0.64,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor=badge_color,
        edgecolor="none",
    )
    banner_ax.add_patch(badge)
    banner_ax.text(8.99, 0.50, badge_text, fontsize=11, fontweight="bold", color="#ffffff", ha="center", va="center")


def make_visual_slide_1():
    """Slide 1: Held-Out Model Performance Bar Chart + Confusion Matrix Heatmap + Test Suite Donut."""
    fig = plt.figure(figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor("#f8fafc")
    _add_slide_banner(
        fig,
        "SLIDE 1",
        "Held-Out Model Verification (2,000 Units) & Automated Testing",
        "99.05% ACCURACY",
        badge_color="#059669",
    )

    # 3 Visual Panels side-by-side
    ax1 = fig.add_axes([0.06, 0.12, 0.27, 0.68])
    ax2 = fig.add_axes([0.39, 0.12, 0.27, 0.68])
    ax3 = fig.add_axes([0.70, 0.12, 0.26, 0.68])

    # Panel 1: Horizontal Metric Bar Chart
    metrics = ["Recall", "F1-Score", "PR-AUC", "Precision", "ROC-AUC", "Accuracy"]
    values = [75.00, 84.30, 88.24, 96.23, 97.50, 99.05]
    colors = ["#f59e0b", "#0d9488", "#6366f1", "#10b981", "#3b82f6", "#1d4ed8"]

    bars = ax1.barh(metrics, values, color=colors, height=0.56, edgecolor="none")
    ax1.set_xlim(0, 113)
    ax1.set_facecolor("#ffffff")
    ax1.set_title("1. Held-Out Test Set Metrics (%)", fontsize=13, fontweight="bold", color="#0f172a", pad=12)
    ax1.set_xlabel("Score (%)", fontsize=10, fontweight="bold", color="#475569")
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    for bar, v in zip(bars, values):
        ax1.text(
            v + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}%",
            va="center",
            ha="left",
            fontsize=10.5,
            fontweight="bold",
            color="#0f172a",
        )

    # Panel 2: 2x2 Confusion Matrix Heatmap
    cm = np.array([[1930, 2], [17, 51]])
    annot = np.array(
        [
            ["TN = 1,930\n(99.9% Normal)", "FP = 2\n(0.1% False Alarm)"],
            ["FN = 17\n(Missed @ t=0.5)", "TP = 51\n(75.0% Caught)"],
        ]
    )
    sns.heatmap(
        np.array([[0.95, 0.08], [0.25, 0.78]]),
        annot=annot,
        fmt="",
        cmap="YlGnBu",
        cbar=False,
        xticklabels=["Predicted:\nNo Failure", "Predicted:\nFailure"],
        yticklabels=["Actual:\nNo Failure", "Actual:\nFailure"],
        annot_kws={"size": 11.5, "weight": "bold"},
        linewidths=3,
        linecolor="#ffffff",
        ax=ax2,
    )
    ax2.set_title("2. Confusion Matrix (2,000 Units)", fontsize=13, fontweight="bold", color="#0f172a", pad=12)
    ax2.tick_params(axis="both", labelsize=10.5)

    # Panel 3: Automated Test Suite Donut Chart (27 / 27 Passing)
    suite_labels = [
        "Milestone 4 E2E\nWorkflow (6/6)",
        "Advanced ML\nTasks (8/8)",
        "Alert Lifecycle\nTests (7/7)",
        "DB & Health\nTests (4/4)",
        "Held-Out Model\nParity (2/2)",
    ]
    suite_counts = [6, 8, 7, 4, 2]
    suite_colors = ["#2563eb", "#7c3aed", "#059669", "#0d9488", "#f59e0b"]

    wedges, texts = ax3.pie(
        suite_counts,
        labels=suite_labels,
        colors=suite_colors,
        startangle=140,
        radius=0.92,
        wedgeprops=dict(width=0.38, edgecolor="#f8fafc", linewidth=2.5),
        textprops=dict(fontsize=9.2, fontweight="bold", color="#1e293b"),
    )
    ax3.text(0, 0.06, "27 / 27", ha="center", va="center", fontsize=20, fontweight="bold", color="#059669")
    ax3.text(0, -0.14, "100% PASS", ha="center", va="center", fontsize=10, fontweight="bold", color="#475569")
    ax3.set_title("3. Automated Test Suite Breakdown", fontsize=13, fontweight="bold", color="#0f172a", pad=12)

    return fig


def make_visual_slide_2():
    """Slide 2: Visual Deployment Pipeline Map + Live Demo Health/Failure Chart + Fleet Risk Donuts."""
    fig = plt.figure(figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor("#f8fafc")
    _add_slide_banner(
        fig,
        "SLIDE 2",
        "Render Cloud Architecture & Live Deployed Telemetry Visuals",
        "LIVE ON RENDER",
        badge_color="#2563eb",
    )

    # Top Panel: Visual Cloud Architecture Flowchart
    flow_ax = fig.add_axes([0.03, 0.56, 0.94, 0.28])
    flow_ax.set_xlim(0, 15)
    flow_ax.set_ylim(0, 3)
    flow_ax.axis("off")

    flow_ax.text(
        0.15,
        2.72,
        "1. Live Render Production Architecture  (https://machine-learning-system-for-early.onrender.com)",
        fontsize=13,
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
        box = patches.FancyBboxPatch(
            (nx, ny),
            nw,
            nh,
            boxstyle="round,pad=0.03,rounding_size=0.18",
            facecolor="#ffffff",
            edgecolor=ncol,
            linewidth=2.0,
        )
        flow_ax.add_patch(box)
        top = patches.FancyBboxPatch(
            (nx, ny + nh - 0.58),
            nw,
            0.58,
            boxstyle="round,pad=0.01,rounding_size=0.14",
            facecolor=ncol,
            edgecolor="none",
        )
        flow_ax.add_patch(top)
        flow_ax.text(nx + nw / 2, ny + nh - 0.29, ntitle, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#ffffff")
        flow_ax.text(nx + nw / 2, ny + 0.68, nsub, ha="center", va="center", fontsize=9.2, color="#334155", linespacing=1.35)

    for ax_x in [2.75, 5.80, 8.85, 11.90]:
        flow_ax.annotate(
            "",
            xy=(ax_x + 0.40, 1.35),
            xytext=(ax_x, 1.35),
            arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.35", color="#475569", lw=2.5),
        )

    # Bottom-Left Panel: Live Demo Observation Trajectory Chart (EQ-RENDER-01)
    ax_traj = fig.add_axes([0.06, 0.09, 0.46, 0.40])
    obs_labels = [
        "Obs 1\n(Normal Start)",
        "Obs 2\n(Steady State)",
        "Obs 3\n(Wear Drift)",
        "Obs 4\n(High Torque)",
        "Obs 5\n(Critical Failure)",
    ]
    health_scores = [99.94, 96.40, 62.15, 28.50, 0.35]
    fail_probs = [0.06, 3.60, 37.85, 71.50, 99.65]
    x_idx = np.arange(len(obs_labels))

    ax_traj.plot(x_idx, health_scores, "o-", color="#059669", linewidth=3, markersize=8, label="Health Score (0–100)")
    ax_traj.plot(x_idx, fail_probs, "s--", color="#e11d48", linewidth=3, markersize=8, label="Failure Probability (%)")
    ax_traj.fill_between(x_idx, health_scores, color="#10b981", alpha=0.08)
    ax_traj.fill_between(x_idx, fail_probs, color="#e11d48", alpha=0.08)

    for i, (hs, fp) in enumerate(zip(health_scores, fail_probs)):
        ax_traj.annotate(f"{hs:.1f}", (x_idx[i], hs), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9, fontweight="bold", color="#047857")
        ax_traj.annotate(f"{fp:.1f}%", (x_idx[i], fp), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=9, fontweight="bold", color="#be123c")

    ax_traj.set_ylim(-12, 115)
    ax_traj.set_xticks(x_idx)
    ax_traj.set_xticklabels(obs_labels, fontsize=9.5, fontweight="bold")
    ax_traj.set_ylabel("Score / Probability (%)", fontsize=10, fontweight="bold")
    ax_traj.set_title("2. Live URL Demo Trajectory (EQ-RENDER-01 Saved History)", fontsize=12.5, fontweight="bold", color="#0f172a", pad=10)
    ax_traj.legend(loc="center left", fontsize=9.5)
    ax_traj.grid(linestyle="--", alpha=0.4)
    for spine in ["top", "right"]:
        ax_traj.spines[spine].set_visible(False)

    # Bottom-Right Panel: Fleet Risk & Health Distribution Donuts
    ax_d1 = fig.add_axes([0.55, 0.12, 0.20, 0.37])
    ax_d2 = fig.add_axes([0.77, 0.12, 0.20, 0.37])

    w1, _ = ax_d1.pie(
        [1840, 107, 53],
        colors=["#10b981", "#f59e0b", "#ef4444"],
        startangle=90,
        radius=0.95,
        wedgeprops=dict(width=0.42, edgecolor="#f8fafc", linewidth=2.5),
    )
    ax_d1.text(0, 0.05, "2,000", ha="center", va="center", fontsize=14, fontweight="bold", color="#0f172a")
    ax_d1.text(0, -0.15, "Fleet Units", ha="center", va="center", fontsize=9, fontweight="bold", color="#475569")
    ax_d1.set_title("3. Fleet Risk Split", fontsize=12, fontweight="bold", color="#0f172a")
    ax_d1.legend(
        w1,
        ["Healthy: 1,840 (92.0%)", "Warning: 107 (5.4%)", "Critical: 53 (2.6%)"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8.2,
        frameon=False,
    )

    w2, _ = ax_d2.pie(
        [1916, 31, 53],
        colors=["#3b82f6", "#f59e0b", "#e11d48"],
        startangle=90,
        radius=0.95,
        wedgeprops=dict(width=0.42, edgecolor="#f8fafc", linewidth=2.5),
    )
    ax_d2.text(0, 0.05, "96.6", ha="center", va="center", fontsize=14, fontweight="bold", color="#0f172a")
    ax_d2.text(0, -0.15, "Avg Health", ha="center", va="center", fontsize=9, fontweight="bold", color="#475569")
    ax_d2.set_title("4. Fleet Health Scores", fontsize=12, fontweight="bold", color="#0f172a")
    ax_d2.legend(
        w2,
        ["Good: 1,916 (95.8%)", "Moderate: 31 (1.6%)", "Poor: 53 (2.6%)"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8.2,
        frameon=False,
    )

    return fig


def make_visual_slide_3():
    """Slide 3: 3 Visual Charts — Threshold PR Curve (T42) + TreeSHAP Importance (T41) + Generalization (T43)."""
    fig = plt.figure(figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor("#f8fafc")
    _add_slide_banner(
        fig,
        "SLIDE 3",
        "Visual Analytics: Threshold Curve (T42), TreeSHAP (T41) & Generalization (T43)",
        "PR-AUC = 88.24%",
        badge_color="#7c3aed",
    )

    ax1 = fig.add_axes([0.05, 0.12, 0.28, 0.68])
    ax2 = fig.add_axes([0.39, 0.12, 0.26, 0.68])
    ax3 = fig.add_axes([0.70, 0.12, 0.27, 0.68])

    # Panel 1: T42 Decision Threshold Curve
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    precision = [77.14, 85.25, 88.14, 92.73, 96.23, 96.00, 97.78, 97.62, 100.0]
    recall = [79.41, 76.47, 76.47, 75.00, 75.00, 70.59, 64.71, 60.29, 48.53]
    f1 = [78.26, 80.62, 81.89, 82.93, 84.30, 81.36, 77.88, 74.55, 65.35]

    ax1.plot(thresholds, precision, "o-", color="#2563eb", linewidth=2.5, label="Precision (%)")
    ax1.plot(thresholds, recall, "s-", color="#e11d48", linewidth=2.5, label="Recall (%)")
    ax1.plot(thresholds, f1, "^-", color="#059669", linewidth=2.8, label="F1-Score (%)")

    ax1.axvline(0.30, color="#f59e0b", linestyle="--", linewidth=1.5, label="Warning (t=0.30)")
    ax1.axvline(0.50, color="#059669", linestyle="--", linewidth=1.5, label="Peak F1 (t=0.50)")
    ax1.axvline(0.70, color="#e11d48", linestyle="--", linewidth=1.5, label="Critical (t=0.70)")

    ax1.set_ylim(42, 104)
    ax1.set_xlabel("Decision Threshold (t)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Metric (%)", fontsize=10, fontweight="bold")
    ax1.set_title("1. Threshold Trade-Off Curve (T42)", fontsize=12.5, fontweight="bold", color="#0f172a", pad=12)
    ax1.legend(loc="lower left", fontsize=8.2)
    ax1.grid(linestyle="--", alpha=0.4)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)

    # Panel 2: T41 Global Exact TreeSHAP Feature Importance
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

    bars2 = ax2.barh(features, shap_shares, color=shap_colors, height=0.55)
    ax2.set_xlim(0, 32)
    ax2.set_xlabel("Share of Total |SHAP| Impact (%)", fontsize=10, fontweight="bold")
    ax2.set_title("2. Exact TreeSHAP Drivers (T41)", fontsize=12.5, fontweight="bold", color="#0f172a", pad=12)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    for b, val in zip(bars2, shap_shares):
        ax2.text(val + 0.6, b.get_y() + b.get_height() / 2, f"{val:.1f}%", va="center", fontsize=9.5, fontweight="bold", color="#0f172a")

    # Panel 3: T43 Cross-Plant Generalization & T38 Multiclass Summary
    plants = ["Plant A\n(Baseline)", "Plant B\n(+3.5K Shift)", "Plant C\n(Heavy Load)", "Plant D\n(Noisy IoT)", "T38 6-Class\nFailure Mode"]
    acc_vals = [99.05, 98.40, 97.60, 98.00, 97.95]
    auc_vals = [97.50, 96.80, 95.90, 96.10, 98.00]
    x = np.arange(len(plants))
    w = 0.34

    b_acc = ax3.bar(x - w / 2, acc_vals, w, label="Accuracy (%)", color="#2563eb")
    b_auc = ax3.bar(x + w / 2, auc_vals, w, label="ROC-AUC / Wt-F1 (%)", color="#10b981")

    for bar in list(b_acc) + list(b_auc):
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax3.set_ylim(85, 102)
    ax3.set_xticks(x)
    ax3.set_xticklabels(plants, fontsize=8.5, fontweight="bold")
    ax3.set_ylabel("Performance (%)", fontsize=10, fontweight="bold")
    ax3.set_title("3. Cross-Plant & Multiclass (T38/T43)", fontsize=12.5, fontweight="bold", color="#0f172a", pad=12)
    ax3.legend(loc="lower right", fontsize=8.5)
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    for spine in ["top", "right"]:
        ax3.spines[spine].set_visible(False)

    return fig


def generate_all():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = REPORTS_DIR / "Member5_Milestone4_Presentation.pdf"

    png_paths = [
        REPORTS_DIR / "m5_m4_slide1_testing_and_model_verification.png",
        REPORTS_DIR / "m5_m4_slide2_render_deployment_and_live_demo.png",
        REPORTS_DIR / "m5_m4_slide3_advanced_tasks_and_documentation.png",
    ]
    makers = [make_visual_slide_1, make_visual_slide_2, make_visual_slide_3]

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

    print("Saved Visual-First 3-Page PDF:", pdf_path)
    print("Saved Visual-First 3-Slide PPTX:", pptx_path)


if __name__ == "__main__":
    generate_all()
