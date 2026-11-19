# Advanced Task T39: Binary vs. Multiclass Classification Comparison Report

## 1. Executive Summary
This report compares **Binary Equipment Failure Classification** (`Failure` vs. `No Failure`) against **Multiclass Specific Failure-Type Classification** (`No Failure`, `HDF`, `OSF`, `PWF`, `TWF`, `RNF`) on the held-out AI4I 2020 test set (**2,000 equipment records**).

| Classification Paradigm | Classes | Accuracy | Weighted F1 | Failure / Macro Precision | Failure / Macro Recall | Failure / Macro F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Binary Classifier (`best_model.joblib`)** | 2 | **99.05%** | **98.99%** | **96.23%** | **75.00%** | **84.30%** | **97.52%** |
| **Multiclass Classifier (`failure_type_model.joblib`)** | 6 | **97.95%** | **98.00%** | **61.46%** | **65.08%** | **62.86%** | **94.73%** |
| **Multiclass Collapsed to Binary (`Any Mode vs. Normal`)** | 2 | **98.15%** | **98.21%** | **70.13%** | **79.41%** | **74.48%** | **95.96%** |

---

## 2. Key Findings & Trade-Off Analysis

1. **Binary Classification Excels at High-Precision Screening**:
   - The binary XGBoost classifier achieves **99.05% overall accuracy**, **96.23% failure precision**, and **84.30% F1-score** because pooling all 68 test failure instances into a single positive class gives the tree booster sufficient sample density (`271` training failures) to learn sharp decision boundaries with only 2 false positives.
2. **Multiclass Classification Provides Actionable Root-Cause Granularity**:
   - Splitting the 339 total dataset failures across 5 specific failure modes (`HDF = 115`, `OSF = 98`, `PWF = 95`, `TWF = 46`, `RNF = 19`) reduces per-class training support (especially for `TWF` and `RNF`, which is inherently random).
   - Despite extreme class sparsity in `RNF`, the multiclass XGBoost model achieves **97.95% overall accuracy** and **98.00% weighted F1**, accurately separating physical failure modes (`HDF` driven by temperature difference + low rotational speed, `OSF` driven by high torque x tool wear, and `PWF` driven by extreme power).
3. **Recommended Production Architecture (Two-Stage Cascade)**:
   - Use the **Binary Classifier** as the primary gatekeeper for **Health Scoring (0–100)** and **Alert Triggering** (maximizing F1 and minimizing false alarms).
   - Use the **Multiclass Failure-Type Model** in parallel to prescribe the **Specific Failure Mode (`HDF`, `OSF`, `PWF`, `TWF`)** and targeted maintenance action whenever risk is elevated.
