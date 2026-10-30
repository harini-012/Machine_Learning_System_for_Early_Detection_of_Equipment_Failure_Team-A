# Evaluation, Model Comparison & Integration Summary

## 1. Executive Overview
This report documents the unified evaluation, model comparison, candidate selection, failure-probability testing, and Remaining Useful Life (RUL) feasibility analysis for the Early Detection of Equipment Failure system. All four candidate classification algorithms trained on the preprocessed and feature-engineered dataset (`8,000` training samples and `2,000` test samples across `9` engineered features) were evaluated under a common validation and testing protocol.

---

## 2. Collected Results from Team Members

### Member 1: Data Preprocessing & Feature Engineering
- **Dataset Split**: Stratified 80/20 train-test split (`random_state=42`) preserving the `3.39%` failure class ratio (`271` training failures, `68` testing failures).
- **Selected Features (`9` columns)**:
  - Sensor measurements: `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`
  - Engineered feature: `Temperature Difference [K]` (`Process temperature [K] - Air temperature [K]`)
  - One-hot encoded product quality variants: `Type_H`, `Type_L`, `Type_M`

### Member 2: Baseline Models (`src/train_baseline_models.py`)
- **Logistic Regression** (`StandardScaler` + `class_weight="balanced"`): Achieved high recall (`0.8235`) but low precision (`0.1425`) due to a large number of false alarms (`337` false positives), resulting in a test F1-score of `0.2430`.
- **Decision Tree** (`max_depth=10`, `class_weight="balanced"`): Improved precision to `0.4808` while maintaining `0.7353` recall (`50` true positives, `54` false positives), yielding a test F1-score of `0.5814`.

### Member 3: Random Forest Model (`src/random_forest_model.py`)
- **Random Forest** (`n_estimators=100`, `max_depth=None`, `random_state=42`): Substantially reduced false alarms (`5` false positives) while detecting `51` of `68` failures (`0.7500` recall, `0.9107` precision), achieving a test F1-score of `0.8226` and ROC-AUC of `0.9669`.
- **Top Feature Importances**: `Torque [Nm]` (`0.3245`), `Rotational speed [rpm]` (`0.2361`), `Temperature Difference [K]` (`0.1488`), and `Tool wear [min]` (`0.1432`).

### Member 4: XGBoost Model (`src/xgboost_model.py`)
- **XGBoost** (`n_estimators=200`, `max_depth=3`, `learning_rate=0.1`, `scale_pos_weight=1.0`): Selected via stratified validation tuning across 16 hyperparameter combinations (`reports/xgboost_tuning_results.csv`). Achieved the highest test precision (`0.9444`), tied for highest tree-based recall (`0.7500`), and achieved the highest overall test F1-score (`0.8361`) and ROC-AUC (`0.9808`).
- **Top Feature Importances**: `Temperature Difference [K]` (`0.2968`), `Torque [Nm]` (`0.1921`), `Rotational speed [rpm]` (`0.1893`), and `Tool wear [min]` (`0.1484`).

---

## 3. Unified Model Comparison Table

All four models were evaluated on a common stratified validation split (`80%` fit / `20%` validation within `X_train`) and on the held-out test set (`X_test`, `2,000` samples: `1,932` Normal, `68` Failure). Full metrics are exported in `reports/model_comparison.csv`:

| Model | Validation F1 | Validation Recall | Validation ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC | TN | FP | FN | TP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **0.7010** | 0.6296 | **0.9776** | **0.9900** | **0.9444** | 0.7500 | **0.8361** | **0.9808** | 1929 | 3 | 17 | 51 |
| **Random Forest** | 0.6966 | 0.5741 | 0.9586 | 0.9890 | 0.9107 | 0.7500 | 0.8226 | 0.9669 | 1927 | 5 | 17 | 51 |
| **Decision Tree** | 0.5152 | 0.6296 | 0.8062 | 0.9640 | 0.4808 | 0.7353 | 0.5814 | 0.8618 | 1878 | 54 | 18 | 50 |
| **Logistic Regression** | 0.2460 | **0.8519** | 0.8926 | 0.8255 | 0.1425 | **0.8235** | 0.2430 | 0.9069 | 1595 | 337 | 12 | 56 |

### Confusion Matrix Plots
- Individual confusion matrices:
  - `reports/logistic_regression_confusion_matrix.png`
  - `reports/decision_tree_confusion_matrix.png`
  - `reports/random_forest_confusion_matrix.png`
  - `reports/xgboost_confusion_matrix.png`
- Combined visual comparisons:
  - `reports/all_models_confusion_matrices.png`
  - `reports/model_comparison_chart.png`

---

## 4. Candidate Selection & Validation Rationale

**Selected Best Model**: **XGBoost (`models/best_model.joblib`)**

1. **Highest Validation Performance**: On the stratified validation split (`src/evaluate_models.py` and `src/tune_model.py`), **XGBoost** achieved the highest validation F1-score (`0.7010` vs. `0.6966` for Random Forest, `0.5152` for Decision Tree, and `0.2460` for Logistic Regression) and the highest validation ROC-AUC (`0.9776`).
2. **Superior Precision-Recall Balance on Test Set**: On the held-out test set, XGBoost achieved a `0.8361` F1-score and `0.9808` ROC-AUC, detecting `51` out of `68` equipment failures (`75.0%` recall) while generating only `3` false alarms out of `1,932` normal operating cycles (`94.44%` precision).
3. **Operational Suitability**: In industrial predictive maintenance, excessive false alarms erode operator trust (as seen with Logistic Regression's `337` false positives), while missed failures cause unplanned downtime. XGBoost provides well-calibrated probabilities that allow flexible threshold adjustment for early-warning alerts.

---

## 5. Failure-Probability Prediction & Early-Warning Integration

The inference pipeline in `src/predict_failure.py` loads `models/best_model.joblib` and `reports/selected_features.csv`, automatically computes `Temperature Difference [K]` and one-hot encodes `Type` when raw sensor inputs are passed, and outputs binary predictions, calibrated failure probabilities (`0.0` to `1.0`), and actionable risk categories:
- **Low Risk** (`failure_probability < 0.30`): Normal operation; routine monitoring.
- **Medium Risk** (`0.30 <= failure_probability < 0.70`): Early warning; schedule preventive inspection.
- **High Risk** (`failure_probability >= 0.70`): Critical alert; immediate maintenance intervention required.

### Test Set Probability Separation
- **Mean predicted failure probability for Normal records (`actual = 0`)**: `0.0115`
- **Mean predicted failure probability for Failure records (`actual = 1`)**: `0.6918`
- **Risk Tier Distribution (`2,000` test records)**: `1,932` Low Risk, `21` Medium Risk, `47` High Risk
- **Sample Scenario Verification**:
  - *Normal Operating Condition*: `failure_probability = 0.0003` (`Low Risk`)
  - *Elevated Tool Wear & High Torque (Overstrain Risk)*: `failure_probability = 0.9600` (`High Risk`)
  - *Low Speed & Small Temp Difference (Heat Dissipation Risk)*: `failure_probability = 0.9714` (`High Risk`)
  - *Extreme Power Overload Condition*: `failure_probability = 0.9830` (`High Risk`)

---

## 6. Remaining Useful Life (RUL) Feasibility Investigation

### 6.1 Dataset Structure Assessment
An investigation of the raw dataset (`10,000` records) was conducted to determine whether supervised **Remaining Useful Life (RUL)** regression (predicting exact time-to-failure in minutes or hours) can be trained directly on the current data:

1. **Cross-Sectional Snapshot Structure (No Longitudinal Asset Trajectories)**:
   - Every row in the dataset has a distinct `Product ID` (`10,000` unique IDs across `10,000` rows).
   - There are no timestamps, sequential cycle counters per machine, or run-to-failure degradation trajectories tracking the same asset from initial installation ($t = 0$) to failure ($t = T_{\text{fail}}$).
   - Because true RUL is defined as $\text{RUL}(t) = T_{\text{fail}} - t$, exact temporal RUL labels cannot be computed without longitudinal run-to-failure logs.

2. **Heterogeneous Failure Modes**:
   - Equipment failures in the dataset arise from five distinct underlying mechanisms:
     - **Heat Dissipation Failure (`HDF`, `115` cases)**: Driven by instantaneous operating conditions (`Temperature Difference [K] < 8.6 K` and `Rotational speed [rpm] < 1380 rpm`).
     - **Overstrain Failure (`OSF`, `98` cases)**: Driven by the product of `Tool wear [min] * Torque [Nm]` exceeding variant limits (`L: 11,000`, `M: 12,000`, `H: 13,000 min·Nm`).
     - **Power Failure (`PWF`, `95` cases)**: Driven by instantaneous mechanical power (`Torque * angular velocity`) outside `[3500 W, 9000 W]`.
     - **Tool Wear Failure (`TWF`, `46` cases)**: Driven by cumulative tool wear reaching the `200–240 min` replacement window.
     - **Random Failure (`RNF`, `19` cases)**: Stochastic failures independent of sensor degradation.
   - Only `TWF` and `OSF` depend directly on cumulative `Tool wear [min]`, whereas `HDF` and `PWF` are acute operational regime failures that can occur at any tool wear stage.

### 6.2 Feasible Proxy RUL Estimation & Path Forward
While direct time-to-failure regression requires longitudinal time-series telemetry, two practical RUL proxy indicators are immediately feasible for system integration:

1. **Remaining Tool-Wear Margin (Minutes to Wear/Overstrain Limit)**:
   - Since tool wear failures occur between `200` and `240` minutes (and overstrain occurs when `Tool wear [min] * Torque [Nm]` exceeds the product threshold), a deterministic **Remaining Tool Allowance** can be calculated at any operating torque $\tau$:
     $$\text{Remaining Wear Margin (min)} = \max\left(0,\; \min\left(200,\; \frac{\text{OverstrainLimit}_{\text{Type}}}{\tau}\right) - \text{Tool wear [min]}\right)$$
   - This gives maintenance operators an immediate estimate of remaining safe tool usage minutes under current load conditions.

2. **Requirements for Full Supervised RUL Regression in Future Releases**:
   - Collect continuous time-stamped sensor streams (`timestamp`, `machine_id`, `cycle_id`) across complete run-to-failure or run-to-replacement cycles.
   - Compute ground-truth countdown targets (`RUL = failure_timestamp - current_timestamp`) and engineer rolling window features (moving averages, degradation slopes, and exponential smoothing over `5`, `15`, and `60` minute windows).
