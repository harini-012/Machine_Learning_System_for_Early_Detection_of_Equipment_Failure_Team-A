"""
Generate an Ultra-Clean, Spacious 3-Page PDF & PPTX Presentation for Member 5:
  - Slide 1 (Summary): Final Stage Team Handoff Strip + 4 Clean Summary Cards of Member 5's Tasks
  - Slide 2 (Visuals 1): 2 Large, Spacious Charts — Held-Out Test Metrics Bar Chart + 2x2 Confusion Matrix Heatmap
  - Slide 3 (Visuals 2): Render Cloud Architecture Flowchart + Live Demo Trajectory Chart & Threshold Curve (T42)
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
    banner_ax = fig.add_axes([0.04, 0.87, 0.92, 0.09])
    banner_ax.set_xlim(0, 10)
    banner_ax.set_ylim(0, 1)
    banner_ax.axis("off")

    bg = patches.FancyBboxPatch(
        (0, 0),
        10,
        1,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor="#0f172a",
        edgecolor="none",
    )
    banner_ax.add_patch(bg)
    banner_ax.text(0.25, 0.50, f"{slide_label}  ·  {title}", fontsize=15.5, fontweight="bold", color="#ffffff", va="center")

    badge = patches.FancyBboxPatch(
        (8.10, 0.18),
        1.70,
        0.64,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor=badge_color,
        edgecolor="none",
    )
    banner_ax.add_patch(badge)
    banner_ax.text(8.95, 0.50, badge_text, fontsize=11, fontweight="bold", color="#ffffff", ha="center", va="center")


def make_slide_1_overview():
    """Slide 1: Clean 4-Card Summary of Final Stage & Member 5 Key Tasks."""
    fig = plt.figure(figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor("#f8fafc")
    _add_slide_banner(
        fig,
        "SLIDE 1",
        "Final Stage Overview & Member 5 Key Tasks",
        "MEMBER 5 ROLE",
        badge_color="#059669",
    )

    ax = fig.add_axes([0.04, 0.05, 0.92, 0.79])
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    # Top Strip: 5-Member Sequential Flow
    strip_card = patches.FancyBboxPatch(
        (0.0, 5.75),
        16.0,
        1.65,
        boxstyle="round,pad=0.02,rounding_size=0.15",
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        linewidth=1.4,
    )
    ax.add_patch(strip_card)
    ax.text(0.35, 7.02, "Milestone 4 Sequential Team Workflow (Members 1 -> 5)", fontsize=13, fontweight="bold", color="#0f172a")

    m_steps = [
        (0.35, "1. Member 1", "Flask & Model (/predict)"),
        (3.50, "2. Member 2", "Dashboard & React UI"),
        (6.65, "3. Member 3", "Equipment & SQL History"),
        (9.80, "4. Member 4", "Alerts, Health & CSVs"),
        (12.95, "5. Member 5 (Me)", "Testing, Deploy & Docs"),
    ]
    for idx, (mx, mtitle, msub) in enumerate(m_steps):
        is_m5 = idx == 4
        mcol = "#059669" if is_m5 else "#2563eb"
        mbg = "#ecfdf5" if is_m5 else "#eff6ff"
        mbox = patches.FancyBboxPatch(
            (mx, 5.95),
            2.70,
            0.82,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=mbg,
            edgecolor=mcol,
            linewidth=2.0 if is_m5 else 1.2,
        )
        ax.add_patch(mbox)
        ax.text(mx + 1.35, 6.50, mtitle, ha="center", va="center", fontsize=11, fontweight="bold", color=mcol)
        ax.text(mx + 1.35, 6.18, msub, ha="center", va="center", fontsize=9.5, color="#334155")
        if idx < 4:
            ax.annotate("", xy=(mx + 3.05, 6.36), xytext=(mx + 2.75, 6.36), arrowprops=dict(arrowstyle="->", color="#64748b", lw=2.2))

    # Bottom 2x2 Grid of 4 Clean Cards for Member 5's Tasks
    cards = [
        (
            0.0,
            2.88,
            "1. End-to-End Automated Testing (27 / 27 Pass)",
            "#2563eb",
            "#eff6ff",
            "• Tested all web pages, Equipment Add/Edit/Delete, & duplicate blocking.\n"
            "• Verified HTTP 400 validation on invalid or negative sensor inputs.\n"
            "• Confirmed saved history persists across reopen + all 3 CSV downloads.",
        ),
        (
            8.2,
            2.88,
            "2. Held-Out Model Verification (2,000 Test Units)",
            "#059669",
            "#ecfdf5",
            "• Accuracy: 99.05%   |   Precision: 96.23% (Only 2 False Alarms)\n"
            "• Recall: 75.00%     |   F1-Score: 84.30%   |   ROC-AUC: 97.50%\n"
            "• Confirmed single-reading Flask service matches batch model output.",
        ),
        (
            0.0,
            0.05,
            "3. Render Cloud Deployment & Live URL Demo",
            "#0284c7",
            "#f0f9ff",
            "• Deployed live: https://machine-learning-system-for-early.onrender.com\n"
            "• Configured Gunicorn + dual PostgreSQL (DATABASE_URL) & SQLite DB.\n"
            "• Ran live demo (EQ-RENDER-01 Normal & High-Risk readings + reopened history).",
        ),
        (
            8.2,
            0.05,
            "4. Documentation (README.md) & Advanced ML (T38–T45)",
            "#7c3aed",
            "#f5f3ff",
            "• Updated README.md with live URL, setup guide, metrics & screenshots.\n"
            "• Analyzed Decision Thresholds (T42: t=0.30 Warning, t=0.50 F1, t=0.70 Critical).\n"
            "• Completed 6-Class Model (T38), Isolation Forest (T40), & TreeSHAP (T41).",
        ),
    ]

    for cx, cy, ctitle, ccol, cbg, cbody in cards:
        cbox = patches.FancyBboxPatch(
            (cx, cy),
            7.8,
            2.52,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor="#ffffff",
            edgecolor="#cbd5e1",
            linewidth=1.4,
        )
        ax.add_patch(cbox)
        top_bar = patches.FancyBboxPatch(
            (cx, cy + 1.92),
            7.8,
            0.60,
            boxstyle="round,pad=0.01,rounding_size=0.12",
            facecolor=cbg,
            edgecolor=ccol,
            linewidth=1.3,
        )
        ax.add_patch(top_bar)
        ax.text(cx + 0.32, cy + 2.22, ctitle, fontsize=12.5, fontweight="bold", color=ccol, va="center")
        ax.text(cx + 0.35, cy + 1.62, cbody, fontsize=10.8, color="#1e293b", va="top", linespacing=1.6)

    return fig


def make_slide_2_model_visuals():
    """Slide 2 (Visuals 1): 2 Large, Spacious Charts — Held-Out Test Metrics + 2x2 Confusion Matrix."""
    fig = plt.figure(figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor("#f8fafc")
    _add_slide_banner(
        fig,
        "SLIDE 2 (VISUALS 1)",
        "Held-Out Test Set Performance & Confusion Matrix (2,000 Units)",
        "99.05% ACCURACY",
        badge_color="#2563eb",
    )

    ax1 = fig.add_axes([0.08, 0.13, 0.40, 0.65])
    ax2 = fig.add_axes([0.56, 0.13, 0.38, 0.65])

    # Left: Large Horizontal Bar Chart of Held-Out Metrics
    metrics = ["Recall", "F1-Score", "PR-AUC", "Precision", "ROC-AUC", "Accuracy"]
    values = [75.00, 84.30, 88.24, 96.23, 97.50, 99.05]
    colors = ["#f59e0b", "#0d9488", "#6366f1", "#10b981", "#3b82f6", "#1d4ed8"]

    bars = ax1.barh(metrics, values, color=colors, height=0.52)
    ax1.set_xlim(0, 114)
    ax1.set_facecolor("#ffffff")
    ax1.set_title("1. Held-Out Test Set Metrics (XGBoost)", fontsize=14.5, fontweight="bold", color="#0f172a", pad=14)
    ax1.set_xlabel("Percentage (%)", fontsize=11.5, fontweight="bold", color="#334155")
    ax1.tick_params(axis="both", labelsize=11.5)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    for bar, v in zip(bars, values):
        ax1.text(
            v + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.2f}%",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#0f172a",
        )

    # Right: Large 2x2 Confusion Matrix Heatmap
    annot = np.array(
        [
            ["TN = 1,930\n\n99.9% Normal\nCorrect", "FP = 2\n\nOnly 2 False\nAlarms (0.1%)"],
            ["FN = 17\n\nMissed Failures\nat t = 0.50", "TP = 51\n\n75.0% True\nFailures Caught"],
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
        annot_kws={"size": 13, "weight": "bold"},
        linewidths=4,
        linecolor="#ffffff",
        ax=ax2,
    )
    ax2.set_title("2. Held-Out Confusion Matrix (2,000 Units)", fontsize=14.5, fontweight="bold", color="#0f172a", pad=14)
    ax2.tick_params(axis="both", labelsize=11.5)

    return fig


def make_slide_3_deployment_visuals():
    """Slide 3 (Visuals 2): Render Cloud Architecture Flowchart + Live Demo Trajectory & Threshold Curve."""
    fig = plt.figure(figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor("#f8fafc")
    _add_slide_banner(
        fig,
        "SLIDE 3 (VISUALS 2)",
        "Render Cloud Architecture, Live Demo Trajectory & Decision Thresholds",
        "LIVE ON RENDER",
        badge_color="#059669",
    )

    # Top Panel: Clean 5-Stage Cloud Architecture Flowchart
    flow_ax = fig.add_axes([0.04, 0.56, 0.92, 0.27])
    flow_ax.set_xlim(0, 15)
    flow_ax.set_ylim(0, 2.8)
    flow_ax.axis("off")

    flow_ax.text(
        0.1,
        2.55,
        "1. Production Cloud Architecture  (https://machine-learning-system-for-early.onrender.com)",
        fontsize=13.5,
        fontweight="bold",
        color="#0f172a",
    )

    nodes = [
        (0.10, 0.25, 2.55, 1.95, "#1e293b", "1. React 18 Web UI", "Prediction Studio\n& Equipment Fleet"),
        (3.20, 0.25, 2.55, 1.95, "#2563eb", "2. Render Cloud", "Gunicorn WSGI\n& Flask App"),
        (6.30, 0.25, 2.55, 1.95, "#7c3aed", "3. ML Engine", "XGBoost Model\n& TreeSHAP"),
        (9.40, 0.25, 2.55, 1.95, "#0d9488", "4. Shared SQL DB", "PostgreSQL / SQLite\n7 Tables (12.2k rows)"),
        (12.45, 0.25, 2.45, 1.95, "#e11d48", "5. Outputs", "Health Score, Alerts\n& 3 CSV Reports"),
    ]
    for nx, ny, nw, nh, ncol, ntitle, nsub in nodes:
        box = patches.FancyBboxPatch((nx, ny), nw, nh, boxstyle="round,pad=0.03,rounding_size=0.16", facecolor="#ffffff", edgecolor=ncol, linewidth=2.0)
        flow_ax.add_patch(box)
        top = patches.FancyBboxPatch((nx, ny + nh - 0.60), nw, 0.60, boxstyle="round,pad=0.01,rounding_size=0.14", facecolor=ncol, edgecolor="none")
        flow_ax.add_patch(top)
        flow_ax.text(nx + nw / 2, ny + nh - 0.30, ntitle, ha="center", va="center", fontsize=11, fontweight="bold", color="#ffffff")
        flow_ax.text(nx + nw / 2, ny + 0.65, nsub, ha="center", va="center", fontsize=10, color="#334155", linespacing=1.4)

    for ax_x in [2.72, 5.82, 8.92, 12.00]:
        flow_ax.annotate("", xy=(ax_x + 0.42, 1.22), xytext=(ax_x, 1.22), arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.35", color="#475569", lw=2.5))

    # Bottom-Left: Live Demo Trajectory Chart (EQ-RENDER-01)
    ax_traj = fig.add_axes([0.06, 0.10, 0.41, 0.39])
    obs_labels = ["Obs 1\n(Normal)", "Obs 2\n(Steady)", "Obs 3\n(Warning)", "Obs 4\n(High Load)", "Obs 5\n(Failure)"]
    health_scores = [99.9, 96.4, 62.2, 28.5, 0.4]
    fail_probs = [0.1, 3.6, 37.8, 71.5, 99.6]
    x_idx = np.arange(len(obs_labels))

    ax_traj.plot(x_idx, health_scores, "o-", color="#059669", linewidth=3, markersize=8, label="Health Score (0–100)")
    ax_traj.plot(x_idx, fail_probs, "s--", color="#e11d48", linewidth=3, markersize=8, label="Failure Probability (%)")
    ax_traj.fill_between(x_idx, health_scores, color="#10b981", alpha=0.08)
    ax_traj.fill_between(x_idx, fail_probs, color="#e11d48", alpha=0.08)

    for i, (hs, fp) in enumerate(zip(health_scores, fail_probs)):
        ax_traj.annotate(f"{hs:.1f}", (x_idx[i], hs), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9.8, fontweight="bold", color="#047857")
        ax_traj.annotate(f"{fp:.1f}%", (x_idx[i], fp), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=9.8, fontweight="bold", color="#be123c")

    ax_traj.set_ylim(-16, 118)
    ax_traj.set_xticks(x_idx)
    ax_traj.set_xticklabels(obs_labels, fontsize=10, fontweight="bold")
    ax_traj.set_title("2. Live URL Demo Trajectory (EQ-RENDER-01 History)", fontsize=13, fontweight="bold", color="#0f172a", pad=10)
    ax_traj.legend(loc="center left", fontsize=9.8)
    ax_traj.grid(linestyle="--", alpha=0.4)
    for spine in ["top", "right"]:
        ax_traj.spines[spine].set_visible(False)

    # Bottom-Right: T42 Decision Threshold Curve
    ax_th = fig.add_axes([0.55, 0.10, 0.40, 0.39])
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    precision = [77.14, 85.25, 88.14, 92.73, 96.23, 96.00, 97.78, 97.62, 100.0]
    recall = [79.41, 76.47, 76.47, 75.00, 75.00, 70.59, 64.71, 60.29, 48.53]
    f1 = [78.26, 80.62, 81.89, 82.93, 84.30, 81.36, 77.88, 74.55, 65.35]

    ax_th.plot(thresholds, precision, "o-", color="#2563eb", linewidth=2.6, label="Precision (%)")
    ax_th.plot(thresholds, recall, "s-", color="#e11d48", linewidth=2.6, label="Recall (%)")
    ax_th.plot(thresholds, f1, "^-", color="#059669", linewidth=2.8, label="F1-Score (%)")
    ax_th.axvline(0.30, color="#f59e0b", linestyle="--", linewidth=1.6, label="Warning Gate (t=0.30)")
    ax_th.axvline(0.50, color="#059669", linestyle="--", linewidth=1.6, label="Peak F1 Gate (t=0.50)")
    ax_th.axvline(0.70, color="#e11d48", linestyle="--", linewidth=1.6, label="Critical Gate (t=0.70)")

    ax_th.set_ylim(44, 104)
    ax_th.set_xlabel("Decision Threshold (t)", fontsize=10, fontweight="bold")
    ax_th.set_ylabel("Score (%)", fontsize=10, fontweight="bold")
    ax_th.set_title("3. Decision Threshold Trade-Off Curve (Task T42)", fontsize=13, fontweight="bold", color="#0f172a", pad=10)
    ax_th.legend(loc="lower left", fontsize=8.8, ncol=2)
    ax_th.grid(linestyle="--", alpha=0.4)
    for spine in ["top", "right"]:
        ax_th.spines[spine].set_visible(False)

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

    print("Saved Ultra-Clean 3-Page PDF:", pdf_path)
    print("Saved Ultra-Clean 3-Slide PPTX:", pptx_path)


if __name__ == "__main__":
    generate_all()
