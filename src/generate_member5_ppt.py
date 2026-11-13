"""
Generate a Visual, High-Impact 3-Slide PowerPoint Presentation for Member 5
(Zero tables — uses custom infographics/charts, big KPI hero cards, and powerful talking points)

Output file: reports/Member5_Visual_Presentation.pptx
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_DIR / "reports"
OUTPUT_PPTX = REPORTS_DIR / "Member5_Visual_Presentation.pptx"

SLIDE1_IMG = REPORTS_DIR / "ppt_slide1_db_architecture.png"
SLIDE2_IMG = REPORTS_DIR / "ppt_slide2_health_and_trends.png"
SLIDE3_IMG = REPORTS_DIR / "ppt_slide3_dashboard_flow.png"

# Executive Dark Theme Colors
BG_DARK = RGBColor(11, 17, 32)         # #0B1120
CARD_BG = RGBColor(21, 31, 50)         # #151F32
CARD_ALT = RGBColor(30, 41, 59)        # #1E293B
BORDER_COL = RGBColor(51, 65, 85)      # #334155
TEXT_WHITE = RGBColor(248, 250, 252)   # #F8FAFC
TEXT_MUTED = RGBColor(148, 163, 184)   # #94A3B8
ACCENT_CYAN = RGBColor(56, 189, 248)   # #38BDF8
ACCENT_INDIGO = RGBColor(99, 102, 241) # #6366F1
ACCENT_GREEN = RGBColor(16, 185, 129)  # #10B981
ACCENT_AMBER = RGBColor(245, 158, 11)  # #F59E0B
ACCENT_RED = RGBColor(239, 68, 68)     # #EF4444


# ===========================================================================
# STEP 1: GENERATE HIGH-DPI VISUAL INFOGRAPHICS FOR EACH SLIDE
# ===========================================================================

def generate_slide1_visual() -> Path:
    """Create a visual architecture + record distribution graphic for Slide 1."""
    fig, (ax_arch, ax_bar) = plt.subplots(
        2, 1, figsize=(7.2, 5.4), dpi=160, gridspec_kw={"height_ratios": [1.15, 1.0]}
    )
    fig.patch.set_facecolor("#151f32")

    # Top subplot: Hub-and-Spoke Architecture Diagram
    ax_arch.set_facecolor("#151f32")
    ax_arch.set_xlim(0, 10)
    ax_arch.set_ylim(0, 5)
    ax_arch.axis("off")
    ax_arch.set_title(
        "How Member 5 Connects All Team Modules",
        color="#f8fafc",
        fontsize=12.5,
        fontweight="bold",
        pad=6,
    )

    nodes = [
        (0.5, 3.3, "Member 1\nML Prediction", "#3b82f6"),
        (2.9, 3.3, "Member 2\nAnomaly & Risk", "#eab308"),
        (5.3, 3.3, "Member 3\nMaintenance", "#10b981"),
        (7.7, 3.3, "Member 4\nAlert Engine", "#f43f5e"),
    ]
    for x, y, label, col in nodes:
        box = mpatches.FancyBboxPatch(
            (x, y), 1.85, 1.25, boxstyle="round,pad=0.12", facecolor="#0f172a", edgecolor=col, linewidth=2
        )
        ax_arch.add_patch(box)
        ax_arch.text(x + 0.925, y + 0.625, label, ha="center", va="center", color="#f8fafc", fontsize=9.5, fontweight="bold")
        ax_arch.annotate(
            "",
            xy=(5.0, 1.85),
            xytext=(x + 0.925, y),
            arrowprops=dict(arrowstyle="->", color=col, lw=2),
        )

    # Central DB Hub
    hub = mpatches.FancyBboxPatch(
        (2.0, 0.3), 6.0, 1.45, boxstyle="round,pad=0.15", facecolor="#1e1b4b", edgecolor="#38bdf8", linewidth=2.5
    )
    ax_arch.add_patch(hub)
    ax_arch.text(
        5.0,
        1.02,
        "Member 5 Shared SQL Database (src/database.py)\nequipment_maintenance_db • 7 Tables • 12,220 Stored Records",
        ha="center",
        va="center",
        color="#38bdf8",
        fontsize=10,
        fontweight="bold",
    )

    # Bottom subplot: Stored SQL Records Chart
    ax_bar.set_facecolor("#0f172a")
    for spine in ax_bar.spines.values():
        spine.set_color("#334155")
    ax_bar.tick_params(colors="#cbd5e1", labelsize=9)

    labels = ["Sensor\nReadings", "ML\nPredictions", "Health\nScores", "Equipment\nRegistry", "Anomaly\nResults", "Maint.\nRecords", "Active\nAlerts"]
    counts = [2020, 2020, 2020, 2000, 2000, 2000, 160]
    colors = ["#38bdf8", "#6366f1", "#10b981", "#8b5cf6", "#f59e0b", "#14b8a6", "#ef4444"]
    bars = ax_bar.bar(labels, counts, color=colors, width=0.58)
    ax_bar.set_title("Records Stored in Shared SQL Database", color="#f8fafc", fontsize=11, fontweight="bold")
    ax_bar.set_ylim(0, 2450)
    for b in bars:
        ax_bar.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 65,
            f"{int(b.get_height()):,}",
            ha="center",
            color="#f8fafc",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(SLIDE1_IMG, facecolor=fig.get_facecolor())
    plt.close(fig)
    return SLIDE1_IMG


def generate_slide2_visual() -> Path:
    """Create a visual Health Score Donut + Historical Trend Bar Chart for Slide 2."""
    fig, (ax_donut, ax_trend) = plt.subplots(1, 2, figsize=(7.4, 4.8), dpi=160)
    fig.patch.set_facecolor("#151f32")

    # Left: Health Score Donut Chart
    ax_donut.set_facecolor("#151f32")
    sizes = [1916, 31, 53]
    labels = ["Good (80–100)\n1,916 (95.8%)", "Moderate (50–79)\n31 (1.6%)", "Poor (<50)\n53 (2.6%)"]
    colors = ["#10b981", "#f59e0b", "#ef4444"]
    wedges, texts = ax_donut.pie(
        sizes,
        labels=labels,
        colors=colors,
        startangle=140,
        wedgeprops=dict(width=0.42, edgecolor="#151f32", linewidth=2),
        textprops=dict(color="#f8fafc", fontsize=8.8, fontweight="bold"),
    )
    ax_donut.text(
        0, 0, "96.55\nAvg Score", ha="center", va="center", color="#38bdf8", fontsize=13, fontweight="bold"
    )
    ax_donut.set_title("Equipment Health Scores\n(Part C: 2,000 Units)", color="#f8fafc", fontsize=11.5, fontweight="bold")

    # Right: Historical Trend Bar Chart
    ax_trend.set_facecolor("#0f172a")
    for spine in ax_trend.spines.values():
        spine.set_color("#334155")
    ax_trend.tick_params(colors="#cbd5e1", labelsize=9.5)

    t_labels = ["Improving\n(Post-Maint.)", "Deteriorating\n(Rising Risk)", "Stable\n(Normal)"]
    t_vals = [7, 53, 1940]
    t_cols = ["#10b981", "#ef4444", "#38bdf8"]
    bars = ax_trend.barh(t_labels, t_vals, color=t_cols, height=0.52)
    ax_trend.set_title("Historical Trend Classification\n(Part D: Multi-Timestamp SQL)", color="#f8fafc", fontsize=11.5, fontweight="bold")
    ax_trend.set_xlim(0, 2350)
    for b in bars:
        ax_trend.text(
            b.get_width() + 45,
            b.get_y() + b.get_height() / 2,
            f"{int(b.get_width()):,} units",
            va="center",
            color="#f8fafc",
            fontsize=9.5,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(SLIDE2_IMG, facecolor=fig.get_facecolor())
    plt.close(fig)
    return SLIDE2_IMG


def generate_slide3_visual() -> Path:
    """Create a visual Streamlit Manual Input Form -> Prediction Banner mockup + Fleet Charts for Slide 3."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.2, 5.3), dpi=160, gridspec_kw={"height_ratios": [1.05, 1.0]}
    )
    fig.patch.set_facecolor("#151f32")

    # Top: Visual Workflow of Streamlit Manual Input Form
    ax_top.set_facecolor("#151f32")
    ax_top.set_xlim(0, 10)
    ax_top.set_ylim(0, 4.2)
    ax_top.axis("off")
    ax_top.set_title(
        "Interactive Streamlit Dashboard Flow (src/streamlit_dashboard.py)",
        color="#f8fafc",
        fontsize=12,
        fontweight="bold",
    )

    steps = [
        (0.3, 0.7, 2.8, 2.7, "1. Manual Sensor Form\n\n• Air Temp: 300.5 K\n• Process Temp: 310.2 K\n• Speed: 1320 rpm\n• Torque: 62.5 Nm\n• Tool Wear: 215 min", "#38bdf8"),
        (3.6, 0.7, 2.8, 2.7, "2. Live AI Evaluation\n\n• M1 Prob: 94.4% Failure\n• M2 Risk: CRITICAL\n• M3 Issue: Tool Wear\n• Action: Replace Tool", "#f59e0b"),
        (6.9, 0.7, 2.8, 2.7, "3. M5 Score & SQL Sync\n\n• Health Score: 5.56/100\n• Status: POOR (<50)\n• Trend: Deteriorating\n• Saved to SQL DB ✓", "#ef4444"),
    ]
    for x, y, w, h, text, col in steps:
        box = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.14", facecolor="#0f172a", edgecolor=col, linewidth=2.2
        )
        ax_top.add_patch(box)
        ax_top.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="#f8fafc", fontsize=8.8, fontweight="bold")

    ax_top.annotate("", xy=(3.55, 2.05), xytext=(3.15, 2.05), arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=2.5))
    ax_top.annotate("", xy=(6.85, 2.05), xytext=(6.45, 2.05), arrowprops=dict(arrowstyle="->", color="#f59e0b", lw=2.5))

    # Bottom: Risk vs Health Comparison Bar Chart
    ax_bot.set_facecolor("#0f172a")
    for spine in ax_bot.spines.values():
        spine.set_color("#334155")
    ax_bot.tick_params(colors="#cbd5e1", labelsize=9)

    categories = ["Healthy Risk\n(1,840)", "Good Health\n(1,916)", "Warning Alerts\n(107)", "Critical Alerts\n(53)", "Unit Tests\n(11 / 11 Passed)"]
    vals = [1840, 1916, 107, 53, 100]
    cols = ["#22c55e", "#10b981", "#f59e0b", "#ef4444", "#6366f1"]
    bars = ax_bot.bar(categories, vals, color=cols, width=0.52)
    ax_bot.set_title("End-to-End Pipeline Verification Highlights", color="#f8fafc", fontsize=11, fontweight="bold")
    ax_bot.set_ylim(0, 2300)
    for idx, b in enumerate(bars):
        label_txt = "100% Pass" if idx == 4 else f"{int(b.get_height()):,}"
        ax_bot.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 60,
            label_txt,
            ha="center",
            color="#f8fafc",
            fontsize=8.8,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(SLIDE3_IMG, facecolor=fig.get_facecolor())
    plt.close(fig)
    return SLIDE3_IMG


# ===========================================================================
# STEP 2: PPTX HELPER FUNCTIONS (NO TABLES — HERO CARDS + POWER POINTS + IMAGES)
# ===========================================================================

def set_bg(slide, color: RGBColor) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_slide_header(slide, pill_text: str, title: str, subtitle: str) -> None:
    top_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.08))
    top_bar.fill.solid()
    top_bar.fill.fore_color.rgb = ACCENT_INDIGO
    top_bar.line.fill.background()

    pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(0.28), Inches(3.3), Inches(0.32))
    pill.fill.solid()
    pill.fill.fore_color.rgb = RGBColor(30, 58, 138)
    pill.line.color.rgb = ACCENT_CYAN
    p = pill.text_frame.paragraphs[0]
    p.text = pill_text
    p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = ACCENT_CYAN

    t_box = slide.shapes.add_textbox(Inches(0.55), Inches(0.65), Inches(12.2), Inches(0.5))
    tp = t_box.text_frame.paragraphs[0]
    tp.text = title
    tp.font.size = Pt(24)
    tp.font.bold = True
    tp.font.color.rgb = TEXT_WHITE

    s_box = slide.shapes.add_textbox(Inches(0.55), Inches(1.16), Inches(12.2), Inches(0.35))
    sp = s_box.text_frame.paragraphs[0]
    sp.text = subtitle
    sp.font.size = Pt(12)
    sp.font.color.rgb = TEXT_MUTED


def add_kpi_card(
    slide, left: float, top: float, width: float, height: float, big_val: str, label: str, accent: RGBColor
) -> None:
    box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    box.adjustments[0] = 0.08
    box.fill.solid()
    box.fill.fore_color.rgb = CARD_BG
    box.line.color.rgb = accent
    box.line.width = Pt(1.5)

    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE

    p1 = tf.paragraphs[0]
    p1.alignment = PP_ALIGN.CENTER
    p1.text = big_val
    p1.font.size = Pt(22)
    p1.font.bold = True
    p1.font.color.rgb = accent

    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    p2.text = label
    p2.font.size = Pt(10)
    p2.font.bold = True
    p2.font.color.rgb = TEXT_MUTED


def add_power_point_card(
    slide, left: float, top: float, width: float, height: float, heading: str, body: str, accent: RGBColor
) -> None:
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    card.adjustments[0] = 0.05
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = BORDER_COL

    # Left accent stripe
    stripe = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left + 0.08), Inches(top + 0.12), Inches(0.08), Inches(height - 0.24)
    )
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = accent
    stripe.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(left + 0.24), Inches(top + 0.08), Inches(width - 0.34), Inches(height - 0.16))
    tf = tb.text_frame
    tf.word_wrap = True

    p1 = tf.paragraphs[0]
    p1.text = heading
    p1.font.size = Pt(13)
    p1.font.bold = True
    p1.font.color.rgb = accent

    p2 = tf.add_paragraph()
    p2.space_before = Pt(3)
    p2.text = body
    p2.font.size = Pt(11)
    p2.font.color.rgb = TEXT_WHITE


# ===========================================================================
# STEP 3: BUILD THE 3 VISUAL SLIDES
# ===========================================================================

def create_visual_slide_1(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, BG_DARK)
    add_slide_header(
        slide,
        "MEMBER 5 • SLIDE 1 OF 3",
        "Shared SQL Database & Team Integration (src/database.py)",
        "One Centralized Database Connecting Predictions, Anomalies, Maintenance & Alerts",
    )

    # 3 Big KPI Hero Cards on top-left
    add_kpi_card(slide, 0.55, 1.62, 1.72, 1.05, "1 Shared DB", "PostgreSQL + SQLite", ACCENT_CYAN)
    add_kpi_card(slide, 2.40, 1.62, 1.72, 1.05, "7 Tables", "Normalized Schema", ACCENT_INDIGO)
    add_kpi_card(slide, 4.25, 1.62, 1.85, 1.05, "12,220", "Total SQL Records", ACCENT_GREEN)

    # 4 Powerful Point Cards on left
    add_power_point_card(
        slide, 0.55, 2.82, 5.55, 0.98,
        "1. Unified Storage Hub (equipment_maintenance_db)",
        "Eliminates isolated CSVs by storing all 2,000 machines, sensor readings, ML predictions, risks, maintenance actions, and alerts in one database.",
        ACCENT_CYAN,
    )
    add_power_point_card(
        slide, 0.55, 3.90, 5.55, 0.98,
        "2. Plug-and-Play Functions for Every Teammate",
        "Provides insert_sensor_reading(), save_prediction(), save_anomaly_and_risk(), save_maintenance_recommendation(), and save_alert().",
        ACCENT_GREEN,
    )
    add_power_point_card(
        slide, 0.55, 4.98, 5.55, 0.98,
        "3. Smart Duplicate Alert Prevention",
        "Automatically blocks duplicate active alerts for the same equipment and failure condition while tracking New → Acknowledged → Resolved states.",
        ACCENT_AMBER,
    )
    add_power_point_card(
        slide, 0.55, 6.06, 5.55, 0.98,
        "4. Fault-Tolerant Connection Management",
        "Uses managed_connection() to auto-commit transactions, roll back safely on errors, and close connections without memory leaks.",
        ACCENT_INDIGO,
    )

    # Right Side Visual Infographic
    img_card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.30), Inches(1.62), Inches(6.48), Inches(5.42))
    img_card.adjustments[0] = 0.03
    img_card.fill.solid()
    img_card.fill.fore_color.rgb = CARD_BG
    img_card.line.color.rgb = BORDER_COL
    slide.shapes.add_picture(str(SLIDE1_IMG), Inches(6.42), Inches(1.72), width=Inches(6.24))


def create_visual_slide_2(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, BG_DARK)
    add_slide_header(
        slide,
        "MEMBER 5 • SLIDE 2 OF 3",
        "Equipment Health Scoring & Historical Trend Analysis",
        "Translating ML Failure Probabilities into Actionable 0–100 Health Scores & Time-Series Trends",
    )

    # Top Formula Banner on Left
    formula_card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(1.62), Inches(5.55), Inches(0.78))
    formula_card.fill.solid()
    formula_card.fill.fore_color.rgb = RGBColor(30, 27, 75)
    formula_card.line.color.rgb = ACCENT_CYAN
    formula_card.line.width = Pt(1.5)
    fp = formula_card.text_frame.paragraphs[0]
    fp.alignment = PP_ALIGN.CENTER
    fp.text = "Health Score  =  100 × (1 − Failure Probability)"
    fp.font.size = Pt(15)
    fp.font.bold = True
    fp.font.color.rgb = ACCENT_CYAN

    # 3 Hero KPI Boxes
    add_kpi_card(slide, 0.55, 2.52, 1.78, 1.0, "96.55 / 100", "Fleet Avg Health", ACCENT_GREEN)
    add_kpi_card(slide, 2.44, 2.52, 1.78, 1.0, "95.8% Good", "1,916 Healthy Units", ACCENT_CYAN)
    add_kpi_card(slide, 4.32, 2.52, 1.78, 1.0, "53 Poor", "Score < 50 (Critical)", ACCENT_RED)

    # 3 Powerful Point Cards
    add_power_point_card(
        slide, 0.55, 3.66, 5.55, 1.05,
        "1. Three-Tier Health Classification (src/health_scoring.py)",
        "• Good (80–100): 1,916 machines in optimal state.\n• Moderate (50–79): 31 machines needing preventive checks.\n• Poor (<50): 53 high-risk machines needing immediate action.",
        ACCENT_GREEN,
    )
    add_power_point_card(
        slide, 0.55, 4.82, 5.55, 1.05,
        "2. Multi-Metric SQL Trend Engine (src/trend_analysis.py)",
        "Queries historical SQL records to track changes (Δ) across all 5 sensors, ML failure probability, health score, and active alerts.",
        ACCENT_CYAN,
    )
    add_power_point_card(
        slide, 0.55, 5.98, 5.55, 1.05,
        "3. Proactive Degradation Detection",
        "Automatically classifies machines into Improving (7 units recovering post-maintenance), Stable (1,940 units), or Deteriorating (53 units).",
        ACCENT_AMBER,
    )

    # Right Side Visual Chart (Donut + Horizontal Trend Bars)
    img_card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.30), Inches(1.62), Inches(6.48), Inches(5.42))
    img_card.adjustments[0] = 0.03
    img_card.fill.solid()
    img_card.fill.fore_color.rgb = CARD_BG
    img_card.line.color.rgb = BORDER_COL
    slide.shapes.add_picture(str(SLIDE2_IMG), Inches(6.40), Inches(2.02), width=Inches(6.28))


def create_visual_slide_3(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(slide, BG_DARK)
    add_slide_header(
        slide,
        "MEMBER 5 • SLIDE 3 OF 3",
        "Interactive Streamlit Dashboard & Manual Sensor Input Form",
        "Real-Time Failure Prediction, Health Scoring & End-to-End Backend Verification",
    )

    # 3 Hero KPI Boxes
    add_kpi_card(slide, 0.55, 1.62, 1.78, 1.05, "5 Modules", "Integrated End-to-End", ACCENT_CYAN)
    add_kpi_card(slide, 2.44, 1.62, 1.78, 1.05, "Live Form", "Streamlit Sensor UI", ACCENT_INDIGO)
    add_kpi_card(slide, 4.32, 1.62, 1.78, 1.05, "11 / 11", "Unit Tests Passed", ACCENT_GREEN)

    # 4 Powerful Point Cards
    add_power_point_card(
        slide, 0.55, 2.82, 5.55, 0.98,
        "1. Manual Sensor Input Form (src/streamlit_dashboard.py)",
        "Operators manually enter Air Temp, Process Temp, Speed, Torque, Tool Wear & Type (L/M/H) or select 1-click failure presets (TWF, HDF, PWF, OSF).",
        ACCENT_CYAN,
    )
    add_power_point_card(
        slide, 0.55, 3.90, 5.55, 0.98,
        "2. Instant Failure Status & Health Score Display",
        "Displays a color-coded FAILURE / NO FAILURE banner, Failure Probability %, 0–100 Health Score progress bar, and abnormal sensor flags.",
        ACCENT_RED,
    )
    add_power_point_card(
        slide, 0.55, 4.98, 5.55, 0.98,
        "3. Live SQL Database Persistence & Alert Control",
        "Every manual prediction is saved directly to the SQL database, and engineers can update alerts from New → Acknowledged → Resolved in one click.",
        ACCENT_GREEN,
    )
    add_power_point_card(
        slide, 0.55, 6.06, 5.55, 0.98,
        "4. Automated Pipeline & Test Suite (src/test_member5.py)",
        "Single-command pipeline (run_milestone3_pipeline.py) verified by 11 automated unit tests covering DB CRUD, health math, and trend logic.",
        ACCENT_INDIGO,
    )

    # Right Side Visual Infographic
    img_card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.30), Inches(1.62), Inches(6.48), Inches(5.42))
    img_card.adjustments[0] = 0.03
    img_card.fill.solid()
    img_card.fill.fore_color.rgb = CARD_BG
    img_card.line.color.rgb = BORDER_COL
    slide.shapes.add_picture(str(SLIDE3_IMG), Inches(6.42), Inches(1.75), width=Inches(6.24))


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    generate_slide1_visual()
    generate_slide2_visual()
    generate_slide3_visual()

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    create_visual_slide_1(prs)
    create_visual_slide_2(prs)
    create_visual_slide_3(prs)

    prs.save(str(OUTPUT_PPTX))
    print(f"Saved visual 3-slide presentation to: {OUTPUT_PPTX}")


if __name__ == "__main__":
    main()
