# Machine Learning System for Early Detection of Equipment Failure (Team A)

An end-to-end **Predictive Maintenance, Anomaly Detection, Alert Management, and Equipment Health Analytics** platform built on the AI4I 2020 Predictive Maintenance dataset (`10,000` total records, `2,000` held-out test records).

---

## Live Deployed Application (Milestone 4)

- **Deployed Production URL**: **https://machine-learning-system-for-early.onrender.com**
- **Live Prediction Dashboard**: https://machine-learning-system-for-early.onrender.com/dashboard
- **Standalone Prediction Test Form**: https://machine-learning-system-for-early.onrender.com/predict
- **Equipment Registry & Management**: https://machine-learning-system-for-early.onrender.com/equipment
- **Sample Live Equipment History (`EQ-RENDER-01` / `OBS-000001`)**: https://machine-learning-system-for-early.onrender.com/equipment/OBS-000001

---

## 1. System Architecture & Key Modules

| Module | File(s) | Core Capabilities |
| :--- | :--- | :--- |
| **ML Failure Prediction** | `src/model_integration.py`, `src/prediction_service.py` | Loads tuned `XGBClassifier` (`models/best_model.joblib`), engineers `Temperature Difference [K]` and one-hot `Type` features, and predicts binary failure & probability. |
| **Anomaly & Risk Engine** | `src/anomaly_detection.py` | Evaluates 5 sensor streams against IQR operating bounds (`Air temp`, `Process temp`, `Speed`, `Torque`, `Tool wear`) and classifies equipment into `Healthy`, `Warning`, or `Critical`. |
| **Maintenance Engine** | `src/maintenance_recommendations.py` | Maps physical sensor anomalies to AI4I failure modes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and generates prioritized maintenance steps & downtime estimates. |
| **Alert Management** | `src/alert_management.py`, `src/run_alerts.py` | Generates deduplicated `Warning` and `Critical` alerts and manages lifecycle transitions (`New` $\rightarrow$ `Acknowledged` $\rightarrow$ `Resolved`). |
| **Shared SQL Database, Health Scoring & Trends** | `src/database.py`, `src/health_scoring.py`, `src/trend_analysis.py`, `sql/schema.sql` | Dual-mode **Cloud PostgreSQL + Shared SQLite** database (`equipment_maintenance_db`) storing `12,220` records across 7 tables, computing $0\text{–}100$ Health Scores and Historical Trends (`Improving`, `Stable`, `Deteriorating`). |
| **Web Application & Dashboards** | `app.py`, `templates/`, `static/`, `src/streamlit_dashboard.py` | Full Flask web application (`/`, `/dashboard`, `/predict`, `/equipment`, `/equipment/<id>`, `/monitoring/<page>`) deployed via Gunicorn on Render, plus interactive Streamlit dashboard. |

---

## 2. Held-Out Test Set Model Performance Verification

Verified on the **2,000 held-out test records** (`data/processed/X_test.csv` and `data/processed/y_test.csv`) using `tests/test_model_performance.py` (saved in `reports/heldout_test_verification.json` & `reports/heldout_test_verification.csv`):

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

## 4. Running the Application Locally

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

---

## 5. Automated Testing & Live Deployment Verification Results

```bash
# Run Milestone 4 End-to-End & Model Performance Tests (8 tests)
python -m unittest discover -s tests -v

# Run Milestone 3 Member 4 & Member 5 Unit Tests (11 tests)
python -m unittest src.test_member5 src.test_alert_management -v
```

### Test Suite Results (`19 / 19` Passing)
- **`tests/test_milestone4_workflow.py` (`6 / 6 OK`)**:
  - `test_1_core_pages_load`: Verified `/`, `/dashboard`, `/predict`, `/equipment`, `/equipment/add`, and `/monitoring/<page>` return HTTP 200.
  - `test_2_equipment_add_edit_and_delete_workflow`: Verified equipment registration, duplicate blocking, invalid type rejection, editing, and deletion.
  - `test_3_invalid_sensor_input_validation`: Verified HTTP 400 error responses for unregistered equipment, non-numeric readings, and negative values.
  - `test_4_prediction_and_history_persistence_across_reopen`: Submitted two readings (`No Failure` and `Failure`), reopened `/equipment/<id>`, and confirmed both readings and predictions persist and trigger `Deteriorating` trend detection.
  - `test_5_alert_status_persistence_and_health_scoring`: Verified health score calculation and alert status persistence (`New` $\rightarrow$ `Acknowledged` $\rightarrow$ `Resolved`).
  - `test_6_report_files_and_download_routes`: Confirmed all report CSVs exist and are populated.
- **`tests/test_model_performance.py` (`2 / 2 OK`)**:
  - Verified held-out test set metrics (`Accuracy = 99.05%`, `F1 = 84.30%`, `ROC-AUC = 97.50%`) and single-reading consistency between `src/prediction_service.py` and `src/model_integration.py`.
- **Live Render URL Demo Verification (`https://machine-learning-system-for-early.onrender.com`)**:
  - Registered `EQ-RENDER-01` via `POST /equipment/add` (`HTTP 200`).
  - Submitted Reading #1 (`298.1 K`, `308.6 K`, `1551 rpm`, `42.8 Nm`, `35 min`) via `POST /dashboard` $\rightarrow$ returned `No Failure` (`HTTP 200`).
  - Submitted Reading #2 (`300.5 K`, `310.2 K`, `1320 rpm`, `62.5 Nm`, `215 min`) via `POST /predict` $\rightarrow$ returned `Failure` (`HTTP 200`).
  - Reopened `GET /equipment/EQ-RENDER-01` (`HTTP 200`) $\rightarrow$ confirmed both saved readings and predictions remain visible in the equipment history table and charts.

---

## 6. Visual Overview & Evaluation Charts

### Fleet Health, Risk, Trend & Database Summary
![Fleet Health, Risk, Trend & Database Overview](reports/member5_dashboard_overview.png)

### Model Comparison & Confusion Matrices
![Model Comparison Chart](reports/model_comparison_chart.png)
![All Models Confusion Matrices](reports/all_models_confusion_matrices.png)
