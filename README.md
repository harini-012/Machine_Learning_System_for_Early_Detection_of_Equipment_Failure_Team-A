# Machine Learning System for Early Detection of Equipment Failure (Team A)

[![Live Demo on Render](https://img.shields.io/badge/Live_Deployed_App-https%3A%2F%2Fmachine--learning--system--for--early.onrender.com-10b981?style=for-the-badge&logo=render)](https://machine-learning-system-for-early.onrender.com)

> **Live Web Application Link**: **https://machine-learning-system-for-early.onrender.com**

An end-to-end **Predictive Maintenance, Anomaly Detection, Alert Management, and Equipment Health Analytics** platform built on the AI4I 2020 Predictive Maintenance dataset (`10,000` total records, `2,000` held-out test records).

---

## Live Deployed Application (Milestone 4)

- **Live Deployed Application (Milestone 4 & Advanced Tasks)**:
  - **Deployed Production URL**: **https://machine-learning-system-for-early.onrender.com**
  - **Live Prediction Dashboard**: https://machine-learning-system-for-early.onrender.com/dashboard
  - **Standalone Prediction Test Form**: https://machine-learning-system-for-early.onrender.com/predict
  - **Equipment Registry & Management**: https://machine-learning-system-for-early.onrender.com/equipment
  - **Advanced ML Tasks (`T38–T42`) Dashboard**: https://machine-learning-system-for-early.onrender.com/advanced
  - **Sample Live Equipment History (`OBS-000001`)**: https://machine-learning-system-for-early.onrender.com/equipment/OBS-000001

---

## Advanced Tasks (`T38` – `T45`)

| Task ID | Advanced Task | Expected Submission | Implementation & Generated Artifacts |
| :---: | :--- | :--- | :--- |
| **T38** | Predict specific failure types instead of only overall failure. | **Failure-type model** | `src/failure_type_model.py`, `models/failure_type_model.joblib`, `models/failure_type_classes.json`, `reports/t38_failure_type_model_report.csv`, `reports/t38_failure_type_confusion_matrix.png` (**97.95% Accuracy**, **98.00% Weighted F1** across 6 classes: `No Failure`, `HDF`, `OSF`, `PWF`, `TWF`, `RNF`) |
| **T39** | Compare binary classification with multiclass classification. | **Comparison report** | `src/binary_vs_multiclass_comparison.py`, `reports/t39_binary_vs_multiclass_comparison.md`, `reports/t39_binary_vs_multiclass_metrics.csv`, `reports/t39_binary_vs_multiclass_chart.png` |
| **T40** | Investigate anomaly detection using Isolation Forest. | **Anomaly detection notebook** | `notebooks/T40_Isolation_Forest_Anomaly_Detection.ipynb`, `src/isolation_forest_anomaly.py`, `reports/t40_isolation_forest_results.csv`, `reports/t40_isolation_forest_comparison.png` |
| **T41** | Explain predictions using feature importance or SHAP. | **Explainability report** | `src/model_explainability.py`, `reports/t41_explainability_report.md`, `reports/t41_feature_importance.csv`, `reports/t41_feature_importance_chart.png` (Exact TreeSHAP `pred_contribs=True` + Permutation F1 Importance + Live local SHAP explanations in `/dashboard`) |
| **T42** | Examine how prediction performance changes at different decision thresholds. | **Precision-recall comparison** | `src/threshold_precision_recall.py`, `reports/t42_precision_recall_threshold_report.md`, `reports/t42_threshold_metrics.csv`, `reports/t42_precision_recall_threshold_chart.png` (`PR-AUC = 88.24%`, threshold analysis across $t \in [0.10, 0.90]$) |
| **T43** | Test the model on a different dataset, if compatible labeled data is available. | **Generalization report** | `src/model_generalization_test.py`, `data/external/external_plant_telemetry.csv` (`1,500` labeled cross-plant observations), `reports/t43_generalization_report.md`, `reports/t43_generalization_metrics.csv`, `reports/t43_generalization_chart.png` |
| **T44** | Design a simulated sensor-monitoring pipeline. | **Architecture diagram and prototype** | `src/sensor_monitoring_pipeline.py`, `reports/t44_pipeline_architecture_diagram.png`, `reports/t44_pipeline_architecture_and_prototype.md`, `reports/t44_simulated_stream_log.csv` (6-stage streaming pipeline + 20-tick progressive degradation simulation) |
| **T45** | Propose how the system could send maintenance alerts. | **Alert design and demo** | `src/maintenance_alert_notifier.py`, `reports/t45_alert_design_and_demo.md`, `reports/t45_alert_dispatch_log.json`, `reports/t45_alert_dispatch_log.csv` (Multi-channel SMTP Email + Slack Block-Kit Webhook + PagerDuty/SMS escalation with deduplication) |

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
