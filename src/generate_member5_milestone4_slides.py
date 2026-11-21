"""
Generate a Clean, Easily Understandable 3-Page PDF Presentation
(and matching PPTX + PNG previews) for Member 5 (Milestone 4).

Outputs:
  - reports/Member5_Milestone4_Presentation.pdf
  - reports/Member5_Milestone4_Demo_Slides.pptx
  - reports/m5_m4_slide1_testing_and_model_verification.png
  - reports/m5_m4_slide2_render_deployment_and_live_demo.png
  - reports/m5_m4_slide3_advanced_tasks_and_documentation.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from pptx import Presentation
from pptx.util import Inches

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_DIR / "reports"


def _setup_slide_canvas():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    return fig, ax


def _draw_top_header(ax, slide_num: str, title: str, subtitle: str, badge_text: str, badge_color="#2563eb"):
    header = patches.FancyBboxPatch(
        (0.6, 7.65),
        14.8,
        0.95,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor="#0f172a",
        edgecolor="#0f172a",
        linewidth=1,
    )
    ax.add_patch(header)
    ax.text(0.95, 8.24, f"{slide_num}  ·  {title}", fontsize=16.5, fontweight="bold", color="#ffffff")
    ax.text(0.95, 7.86, subtitle, fontsize=11, color="#cbd5e1")

    badge = patches.FancyBboxPatch(
        (12.55, 7.88),
        2.55,
        0.5,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=badge_color,
        edgecolor="none",
    )
    ax.add_patch(badge)
    ax.text(13.825, 8.13, badge_text, fontsize=11, fontweight="bold", color="#ffffff", ha="center", va="center")


def _draw_white_card(ax, x, y, w, h, title, accent="#2563eb"):
    card = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        linewidth=1.4,
    )
    ax.add_patch(card)
    bar = patches.FancyBboxPatch(
        (x, y + h - 0.12),
        w,
        0.12,
        boxstyle="round,pad=0.0,rounding_size=0.06",
        facecolor=accent,
        edgecolor="none",
    )
    ax.add_patch(bar)
    ax.text(x + 0.35, y + h - 0.48, title, fontsize=14, fontweight="bold", color="#0f172a", va="top")


def make_slide_1():
    """Slide 1: End-to-End Automated Testing & Held-Out Model Verification."""
    fig, ax = _setup_slide_canvas()
    _draw_top_header(
        ax,
        "SLIDE 1",
        "Automated Workflow Testing & Held-Out Model Verification",
        "Member 5 (Milestone 4)  ·  Files: tests/test_milestone4_workflow.py  |  tests/test_model_performance.py",
        "27 / 27 TESTS PASS",
        badge_color="#059669",
    )

    # Left Card: What We Tested
    _draw_white_card(ax, 0.6, 0.65, 7.2, 6.65, "1. Complete Workflow Testing (6 Core Areas)", accent="#2563eb")

    items = [
        ("1. All Application Pages Load", "Verified /, /dashboard, /predict, /equipment, and /monitoring/* return HTTP 200."),
        ("2. Equipment Management (CRUD)", "Tested Add, Edit, and Delete equipment + blocked duplicate IDs."),
        ("3. Invalid Sensor Input Handling", "Confirmed HTTP 400 error messages for text, negative, or unregistered inputs."),
        ("4. Prediction & History Persistence", "Submitted 2 readings and reopened /equipment/<id> to verify saved history."),
        ("5. Alert Status Lifecycle", "Verified alert status updates (New -> Acknowledged -> Resolved) persist in DB."),
        ("6. CSV Report Downloads", "Confirmed Predicted Failures, Maintenance, and Model Performance CSV exports."),
    ]
    y = 6.1
    for heading, sub in items:
        box = patches.FancyBboxPatch(
            (0.92, y - 0.68),
            6.56,
            0.78,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor="#f8fafc",
            edgecolor="#e2e8f0",
            linewidth=1,
        )
        ax.add_patch(box)
        ax.text(1.12, y - 0.18, f"✓  {heading}", fontsize=11.2, fontweight="bold", color="#1e40af")
        ax.text(7.28, y - 0.18, "PASSED", fontsize=9.5, fontweight="bold", color="#059669", ha="right")
        ax.text(1.12, y - 0.48, sub, fontsize=9.6, color="#475569")
        y -= 0.92

    # Right Card: Held-Out Test Set Results
    _draw_white_card(ax, 8.15, 0.65, 7.25, 6.65, "2. Held-Out Test Set Results (2,000 Test Records)", accent="#059669")

    kpis = [
        (8.48, 4.95, "ACCURACY", "99.05%", "1,981 / 2,000 Correct", "#2563eb", "#eff6ff"),
        (11.92, 4.95, "PRECISION", "96.23%", "Only 2 False Alarms", "#059669", "#ecfdf5"),
        (8.48, 3.35, "F1-SCORE", "84.30%", "Recall: 75.00% (51/68)", "#d97706", "#fffbeb"),
        (11.92, 3.35, "ROC-AUC", "97.50%", "Strong Class Separation", "#7c3aed", "#f5f3ff"),
    ]
    for kx, ky, klabel, kval, ksub, kcol, kbg in kpis:
        kbox = patches.FancyBboxPatch(
            (kx, ky),
            3.15,
            1.35,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=kbg,
            edgecolor=kcol,
            linewidth=1.3,
        )
        ax.add_patch(kbox)
        ax.text(kx + 1.575, ky + 1.08, klabel, fontsize=9.8, fontweight="bold", color=kcol, ha="center")
        ax.text(kx + 1.575, ky + 0.56, kval, fontsize=22, fontweight="bold", color="#0f172a", ha="center")
        ax.text(kx + 1.575, ky + 0.18, ksub, fontsize=9.2, color="#475569", ha="center")

    # Confusion Matrix Summary Box
    cm_box = patches.FancyBboxPatch(
        (8.48, 0.98),
        6.59,
        2.08,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor="#f8fafc",
        edgecolor="#cbd5e1",
        linewidth=1.2,
    )
    ax.add_patch(cm_box)
    ax.text(8.72, 2.75, "Confusion Matrix & Model Verification (models/best_model.joblib)", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax.text(
        8.72,
        2.32,
        "• True Negatives (TN): 1,930   |   False Positives (FP): 2   (99.9% normal alarm-free)\n"
        "• True Positives (TP): 51      |   False Negatives (FN): 17  (68 total test failures)\n"
        "• Verified single-reading Flask predictions match batch model predictions identically.",
        fontsize=9.6,
        color="#334155",
        va="top",
        linespacing=1.55,
    )

    return fig


def make_slide_2():
    """Slide 2: Cloud Deployment on Render & Live URL Demo Verification."""
    fig, ax = _setup_slide_canvas()
    _draw_top_header(
        ax,
        "SLIDE 2",
        "Cloud Deployment on Render & Live URL Demo",
        "Live URL: https://machine-learning-system-for-early.onrender.com  ·  Files: Procfile | render.yaml | requirements.txt",
        "LIVE ON RENDER",
        badge_color="#2563eb",
    )

    # Left Card: How It Was Deployed
    _draw_white_card(ax, 0.6, 0.65, 7.2, 6.65, "1. Cloud Deployment Setup (Render + Database)", accent="#0284c7")

    deploy_boxes = [
        ("Flask + Gunicorn Production Server", "Configured Procfile & render.yaml (web: gunicorn app:app) with\ntrained XGBoost model, selected features, templates, and static files."),
        ("Shared SQL Database (DATABASE_URL)", "Connected to PostgreSQL via DATABASE_URL with automatic SQLite\nfallback storing 12,220 records across 7 normalized tables."),
        ("Fast Query & Endpoint Optimization", "Optimized monitoring queries to single indexed SQL statements (<10 ms)\nand added GET /health production health-check endpoint."),
        ("Complete README.md Documentation", "Added live deployed Render link, setup instructions, architecture\noverview, held-out test metrics, and verification screenshots."),
    ]
    y = 6.05
    for dtitle, dsub in deploy_boxes:
        box = patches.FancyBboxPatch(
            (0.92, y - 1.02),
            6.56,
            1.15,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor="#f8fafc",
            edgecolor="#e2e8f0",
            linewidth=1,
        )
        ax.add_patch(box)
        ax.text(1.15, y - 0.22, f"⚙  {dtitle}", fontsize=11.5, fontweight="bold", color="#0369a1")
        ax.text(1.15, y - 0.52, dsub, fontsize=9.8, color="#334155", va="top", linespacing=1.4)
        y -= 1.35

    # Right Card: Live Demo Verification Steps
    _draw_white_card(ax, 8.15, 0.65, 7.25, 6.65, "2. Live Demo Verification on Deployed URL", accent="#059669")

    steps = [
        ("STEP 1 · Register Equipment (/equipment/add)", "#2563eb", "#eff6ff", "Added machine EQ-RENDER-01 (Type L, Product L47181) on the live site.\nConfirmed it appears immediately at the top of the Equipment Registry."),
        ("STEP 2 · Submit Normal Sensor Reading (/dashboard)", "#059669", "#ecfdf5", "Input: 298.1 K, 308.6 K, 1551 rpm, 42.8 Nm, 25 min\nResult: No Failure (0.06% Prob)  |  Health Score: 99.94 / 100 (Good)"),
        ("STEP 3 · Submit High-Risk Reading (/dashboard)", "#dc2626", "#fef2f2", "Input: 300.5 K, 310.2 K, 1320 rpm, 62.5 Nm, 215 min\nResult: Failure (99.65% Prob)  |  Health: 0.35 (Poor)  |  Critical Alert + Maintenance Plan"),
        ("STEP 4 · Reopen History & Download 3 CSV Reports", "#7c3aed", "#f5f3ff", "Reopened /equipment/EQ-RENDER-01 -> Both readings & charts stayed visible.\nDownloaded Predicted Failures, Maintenance, and Model Performance CSVs."),
    ]
    y = 6.05
    for stitle, scol, sbg, ssub in steps:
        box = patches.FancyBboxPatch(
            (8.48, y - 1.02),
            6.59,
            1.15,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=sbg,
            edgecolor=scol,
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(8.72, y - 0.22, stitle, fontsize=11.0, fontweight="bold", color=scol)
        ax.text(8.72, y - 0.52, ssub, fontsize=9.5, color="#1e293b", va="top", linespacing=1.4)
        y -= 1.35

    return fig


def make_slide_3():
    """Slide 3: Decision Threshold Analysis (T42) & Advanced ML Suite (T38-T45)."""
    fig, ax = _setup_slide_canvas()
    _draw_top_header(
        ax,
        "SLIDE 3",
        "Decision Threshold Analysis (T42) & Advanced ML Tasks",
        "Live Explorer: https://machine-learning-system-for-early.onrender.com/advanced  ·  Tasks T38 through T45",
        "PR-AUC = 88.24%",
        badge_color="#7c3aed",
    )

    # Left Card: T42 Decision Threshold & Precision-Recall Comparison
    _draw_white_card(ax, 0.6, 0.65, 7.2, 6.65, "1. Decision Threshold Analysis (Task T42)", accent="#7c3aed")

    ax.text(
        0.95,
        6.15,
        "How prediction performance changes across decision thresholds (2,000 test units):",
        fontsize=10.5,
        color="#334155",
    )

    t_cards = [
        ("Threshold t = 0.30  (Warning Alert Gate)", "#d97706", "#fffbeb", "Precision: 88.14%   |   Recall: 76.47%   |   F1: 81.89%\nCatches 52 / 68 failures early -> Triggers preventive Warning inspection."),
        ("Threshold t = 0.50  (Default Prediction Gate)", "#2563eb", "#eff6ff", "Precision: 96.23%   |   Recall: 75.00%   |   F1: 84.30% (Peak F1)\nBest overall balance -> Only 2 false alarms across 1,932 normal machines."),
        ("Threshold t = 0.70  (Critical Shutdown Gate)", "#dc2626", "#fef2f2", "Precision: 97.78%   |   Recall: 64.71%   |   F1: 77.88%\nNear-zero false alarms (1 FP) -> Justifies immediate machine shutdown."),
    ]
    y = 5.82
    for ttitle, tcol, tbg, tsub in t_cards:
        box = patches.FancyBboxPatch(
            (0.92, y - 0.98),
            6.56,
            1.08,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=tbg,
            edgecolor=tcol,
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(1.15, y - 0.18, ttitle, fontsize=11.0, fontweight="bold", color=tcol)
        ax.text(1.15, y - 0.48, tsub, fontsize=9.5, color="#1e293b", va="top", linespacing=1.38)
        y -= 1.26

    summary_box = patches.FancyBboxPatch(
        (0.92, 0.92),
        6.56,
        1.02,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor="#f8fafc",
        edgecolor="#cbd5e1",
        linewidth=1,
    )
    ax.add_patch(summary_box)
    ax.text(1.15, 1.68, "Key Takeaway: Why We Use a Multi-Threshold Policy", fontsize=10.5, fontweight="bold", color="#0f172a")
    ax.text(1.15, 1.36, "Combining t = 0.30 (Warning) and t = 0.70 (Critical) maximizes early failure capture\nwhile keeping emergency shutdown precision above 97.7%.", fontsize=9.4, color="#475569", va="top")

    # Right Card: Complete Advanced Suite (T38-T45)
    _draw_white_card(ax, 8.15, 0.65, 7.25, 6.65, "2. Additional Advanced Tasks Completed (T38–T45)", accent="#d97706")

    adv_list = [
        ("T38 & T39 · Specific Failure-Type Model & Comparison", "Trained 6-class failure model (No Failure, HDF, OSF, PWF, TWF, RNF)\nachieving 97.95% Accuracy & 98.00% Weighted F1 + comparison report."),
        ("T40 & T41 · Isolation Forest & TreeSHAP Explainability", "Created Isolation Forest Notebook (~82% Anomaly ROC-AUC) & Exact\nTreeSHAP top feature drivers (Torque, Speed, Tool Wear, ΔT)."),
        ("T43 · Cross-Dataset Generalization Report", "Evaluated model on 1,500 external labeled samples across 3 shifted\nplant environments (+3.5K Thermal, Heavy Load, Noisy Sensors >95% AUC)."),
        ("T44 & T45 · Sensor Stream Pipeline & Alert Dispatcher", "Built 6-stage 20-tick sensor streaming prototype + Multi-Channel Alert\nDispatcher (SMTP Email Work Order + Slack Webhook + PagerDuty/SMS)."),
    ]
    y = 6.05
    for atitle, asub in adv_list:
        box = patches.FancyBboxPatch(
            (8.48, y - 1.02),
            6.59,
            1.15,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor="#f8fafc",
            edgecolor="#e2e8f0",
            linewidth=1,
        )
        ax.add_patch(box)
        ax.text(8.72, y - 0.22, f"★  {atitle}", fontsize=11.0, fontweight="bold", color="#b45309")
        ax.text(8.72, y - 0.52, asub, fontsize=9.5, color="#334155", va="top", linespacing=1.4)
        y -= 1.35

    return fig


def generate_all():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = REPORTS_DIR / "Member5_Milestone4_Presentation.pdf"

    png_paths = [
        REPORTS_DIR / "m5_m4_slide1_testing_and_model_verification.png",
        REPORTS_DIR / "m5_m4_slide2_render_deployment_and_live_demo.png",
        REPORTS_DIR / "m5_m4_slide3_advanced_tasks_and_documentation.png",
    ]
    makers = [make_slide_1, make_slide_2, make_slide_3]

    with PdfPages(pdf_path) as pdf:
        for maker, png_path in zip(makers, png_paths):
            fig = maker()
            plt.tight_layout()
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            fig.savefig(png_path, dpi=160, facecolor=fig.get_facecolor())
            plt.close(fig)

    # Also update the PPTX deck with the clean light slides
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]
    for img_path in png_paths:
        slide = prs.slides.add_slide(blank_layout)
        slide.shapes.add_picture(str(img_path), Inches(0), Inches(0), width=Inches(13.333), height=Inches(7.5))
    pptx_path = REPORTS_DIR / "Member5_Milestone4_Demo_Slides.pptx"
    prs.save(str(pptx_path))

    print("Saved 3-Page Clean PDF Presentation:", pdf_path)
    print("Saved 3-Slide PPTX Presentation:", pptx_path)


if __name__ == "__main__":
    generate_all()
