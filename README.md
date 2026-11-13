# Machine Learning System for Early Detection of Equipment Failure (Team A)

An end-to-end **Predictive Maintenance, Anomaly Detection, Alert Management, and Equipment Health Analytics** platform built on the AI4I 2020 Predictive Maintenance dataset (`10,000` records, `2,000` held-out test records).

---

## 1. System Architecture & Key Features

| Module | File(s) | Core Capabilities |
| :--- | :--- | :--- |
| **ML Failure Prediction** | `src/model_integration.py`, `src/prediction_service.py` | Loads tuned `XGBClassifier` (`models/best_model.joblib`), engineers `Temperature Difference [K]` and one-hot `Type` features, and predicts binary failure & probability. |
| **Anomaly & Risk Engine** | `src/anomaly_detection.py` | Evaluates 5 sensor streams against IQR operating bounds (`Air temp`, `Process temp`, `Speed`, `Torque`, `Tool wear`) and classifies equipment into `Healthy`, `Warning`, or `Critical`. |
| **Maintenance Engine** | `src/maintenance_recommendations.py` | Maps physical sensor anomalies to AI4I failure modes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and generates prioritized maintenance steps & downtime estimates. |
| **Alert Management** | `src/alert_management.py`, `src/run_alerts.py` | Generates deduplicated `Warning` and `Critical` alerts and manages lifecycle transitions (`New` $\rightarrow$ `Acknowledged` $\rightarrow$ `Resolved`). |
| **Shared SQL Database, Health Scoring & Trends** | `src/database.py`, `src/health_scoring.py`, `src/trend_analysis.py`, `sql/schema.sql` | Dual-mode **Cloud PostgreSQL + Shared SQLite** database (`equipment_maintenance_db`) storing `12,220` records across 7 tables, computing $0\text{–}100$ Health Scores and Historical Trends (`Improving`, `Stable`, `Deteriorating`). |
| **Web Application & Dashboards** | `app.py`, `templates/`, `static/`, `src/streamlit_dashboard.py` | Full Flask web application (`/`, `/dashboard`, `/predict`, `/equipment`, `/equipment/<id>`, `/monitoring/<page>`) and interactive Streamlit dashboard. |

---

## 2. Held-Out Test Set Model Performance Verification

Verified on the **2,000 held-out test records** (`data/processed/X_test.csv` and `data/processed/y_test.csv`) using `tests/test_model_performance.py` (saved in `reports/heldout_test_verification.json`):

| Metric | Held-Out Test Result (`XGBClassifier`) |
| :--- | ---: |
| **Test Set Size** | `2,000` records (`68` actual failures) |
| **Accuracy** | **99.05%** (`0.9905`) |
| **Precision** | **96.23%** (`0.9623`) |
| **Recall** | **75.00%** (`0.7500`) |
| **F1-Score** | **84.30%** (`0.8430`) |
| **ROC-AUC** | **97.50%** (`0.9750`) |
| **Confusion Matrix** | `TN = 1,930` \| `FP = 2` \| `FN = 17` \| `TP = 51` |

---

## 3. Setup & Installation Instructions

### Prerequisites
- Python `3.11+`
- Git

### 1. Clone the Repository & Install Dependencies
```bash
git clone https://github.com/harini-012/Machine_Learning_System_for_Early_Detection_of_Equipment_Failure_Team-A.git
cd Machine_Learning_System_for_Early_Detection_of_Equipment_Failure_Team-A

python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Database Configuration (`DATABASE_URL`)
- **Default Zero-Config Mode (Shared SQLite)**:
  If `DATABASE_URL` is not set, `src/database.py` automatically connects to `data/equipment_maintenance_db.sqlite` (pre-populated with all 7 tables and 2,000 equipment records).
- **Cloud PostgreSQL Mode (Neon / Supabase / Render)**:
  Copy `.env.example` to `.env` and set your PostgreSQL URI:
  ```env
  DATABASE_URL=postgresql://user:password@host/equipment_maintenance_db?sslmode=require
  ```

---

## 4. Running the Application

### Option A: Run the Flask Web Application (`app.py`)
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser:
- `/`: Home overview
- `/dashboard`: Main prediction dashboard (select registered equipment, input 5 sensor values, view failure probability & status, and save observation to database)
- `/predict`: Standalone prediction test form
- `/equipment`: Equipment registry list (`Add`, `Edit`, `View History`, `Delete`)
- `/equipment/<equipment_id>`: Equipment historical sensor & prediction charts (`Chart.js`) and tables

### Option B: Run the Interactive Streamlit Dashboard
```bash
streamlit run src/streamlit_dashboard.py
```
Open **http://localhost:8501** in your browser.

### Option C: Run the End-to-End Milestone 3 Backend Pipeline
```bash
python -m src.run_milestone3_pipeline
```

---

## 5. Automated Testing & Verification Suite

Run the complete test suite covering Flask routes, invalid input handling, equipment CRUD, prediction history persistence across page reopens, alert status transitions, health scoring, trend analysis, and held-out model evaluation:

```bash
# Run Milestone 4 End-to-End & Model Performance Tests (8 tests)
python -m unittest discover -s tests -v

# Run Milestone 3 Member 4 & Member 5 Unit Tests (11 tests)
python -m unittest src.test_member5 src.test_alert_management -v
```

### Latest Test Execution Summary
- `tests/test_milestone4_workflow.py`: **6 / 6 passed** (Core routes, Equipment Add/Edit/Delete, Invalid Input HTTP 400 validation, Multi-observation History Persistence, Alert Lifecycle & Health Scoring, Report Verification)
- `tests/test_model_performance.py`: **2 / 2 passed** (Held-out 2,000-record evaluation & single-reading service consistency)
- `src/test_member5.py` & `src/test_alert_management.py`: **11 / 11 passed**

---

## 6. Cloud Deployment Instructions (Render / Railway)

The repository includes `Procfile` and `render.yaml` for one-click cloud deployment:

1. Push the repository to GitHub.
2. In **Render** (https://dashboard.render.com) or **Railway**:
   - Create a new **Web Service** connected to this repository.
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Environment Variable**: Set `DATABASE_URL` to your shared Cloud PostgreSQL connection string (or leave unset to use the bundled SQLite database).
3. Once deployed, open the live URL, register or select an equipment unit under `/equipment`, submit two readings under `/dashboard`, and reopen `/equipment/<equipment_id>` to verify persistent history.

---

## 7. Dashboard & Evaluation Visuals

- **Fleet Health, Risk, Trend & Database Summary**: `reports/member5_dashboard_overview.png`
- **Model Comparison Chart**: `reports/model_comparison_chart.png`
- **Confusion Matrices**: `reports/all_models_confusion_matrices.png`
