# Advanced Task T42: Precision-Recall & Decision Threshold Comparison Report

## 1. Executive Summary
Standard binary classification applies a fixed decision threshold of $t = 0.50$. However, in industrial predictive maintenance, the cost of an **unplanned catastrophic breakdown (`False Negative`)** is typically 5x–20x higher than the cost of a **preventive technician inspection (`False Positive`)**.

We evaluated the production `XGBClassifier` (`models/best_model.joblib`) across 11 decision thresholds $t \in [0.10, 0.90]$ on the **2,000 held-out test records (`68` true failures)**.
- **Precision-Recall AUC (`Average Precision`)**: **`87.81%` (`0.8781`)**
- **Optimal Balanced F1 Threshold**: **`t = 0.50`** (`F1 = 84.30%`, `Precision = 96.23%`, `Recall = 75.00%`)
- **Optimal Recall-Weighted F2 Threshold**: **`t = 0.25`** (`F2 = 82.61%`, `Recall = 83.82%`)

---

## 2. Threshold Performance Comparison Table

| Threshold ($t$) | Precision | Recall | F1-Score | F2-Score | TP (Caught) | FP (False Alarms) | FN (Missed) | Operational Regime |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.10** | `53.10%` | `88.24%` | **66.30%** | `77.92%` | `60` | `53` | `8` | Aggressive Early Screening |
| **0.15** | `60.82%` | `86.76%` | **71.52%** | `79.95%` | `59` | `38` | `9` | Aggressive Early Screening |
| **0.20** | `69.05%` | `85.29%` | **76.32%** | `81.46%` | `58` | `26` | `10` | Aggressive Early Screening |
| **0.25** | `78.08%` | `83.82%` | **80.85%** | `82.61%` | `57` | `16` | `11` | Aggressive Early Screening |
| **0.30** | `79.71%` | `80.88%` | **80.29%** | `80.65%` | `55` | `14` | `13` | Warning Alert Threshold (High Recall) |
| **0.40** | `89.83%` | `77.94%` | **83.46%** | `80.06%` | `53` | `6` | `15` | Conservative High-Confidence |
| **0.50** | `96.23%` | `75.00%` | **84.30%** | `78.46%` | `51` | `2` | `17` | Default Classification Threshold (Max F1) |
| **0.60** | `97.92%` | `69.12%` | **81.03%** | `73.44%` | `47` | `1` | `21` | Conservative High-Confidence |
| **0.70** | `97.67%` | `61.76%` | **75.68%** | `66.67%` | `42` | `1` | `26` | Critical Alert Threshold (Max Precision) |
| **0.80** | `100.00%` | `48.53%` | **65.35%** | `54.10%` | `33` | `0` | `35` | Conservative High-Confidence |
| **0.90** | `100.00%` | `29.41%` | **45.45%** | `34.25%` | `20` | `0` | `48` | Conservative High-Confidence |

---

## 3. Why Our System Uses a Multi-Threshold Risk Policy ($t = 0.30$ and $t = 0.70$)

1. **Warning Threshold at $t = 0.30$ (High-Recall Early Warning)**:
   - Lowering the decision threshold from $0.50$ to $0.30$ increases **Failure Recall** while keeping false alarms manageable, ensuring subtle early-stage degradation is flagged as a `Warning` alert before tool failure occurs.
2. **Default Prediction Threshold at $t = 0.50$ (Peak F1 Balance)**:
   - Achieves **96.23% Precision** and **75.00% Recall** (`F1 = 84.30%`), generating only **2 false positives** across 2,000 machines.
3. **Critical Shutdown Threshold at $t = 0.70$ (Ultra-High Precision)**:
   - At $t \ge 0.70$, **Precision exceeds 97%+**, justifying immediate machine shutdown and emergency maintenance dispatch with near-zero false-alarm downtime.
