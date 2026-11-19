# Advanced Task T41: Model Explainability Report (Exact TreeSHAP & Feature Importance)

## 1. Methodology
To explain how the production `XGBClassifier` (`models/best_model.joblib`) predicts equipment failures, we computed three complementary explainability metrics across all **2,000 held-out test observations**:
1. **Exact TreeSHAP (`Mean |SHAP|` in log-odds)**: Computed via XGBoost's exact tree path contribution algorithm (`pred_contribs=True`), measuring each feature's average absolute marginal contribution to the predicted failure log-odds.
2. **XGBoost Gain Importance**: Average training loss reduction achieved whenever a feature is used to split a decision node.
3. **Permutation F1 Drop**: Decrease in held-out test set F1-score when a feature's values are randomly shuffled across 10 repetitions.

---

## 2. Global Feature Importance Ranking

| Rank | Feature | Mean \|SHAP\| (Log-Odds) | SHAP Impact Share (%) | XGBoost Gain Importance | Permutation F1 Drop |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | **Torque [Nm]** | `1.2314` | `34.38%` | `0.2136` | `0.4259` |
| 2 | **Tool wear [min]** | `1.2086` | `33.74%` | `0.1547` | `0.1958` |
| 3 | **Temperature Difference [K]** | `0.4448` | `12.42%` | `0.3058` | `0.3586` |
| 4 | **Rotational speed [rpm]** | `0.4173` | `11.65%` | `0.1685` | `0.3403` |
| 5 | **Process temperature [K]** | `0.1095` | `3.06%` | `0.0226` | `0.0108` |
| 6 | **Air temperature [K]** | `0.0964` | `2.69%` | `0.0318` | `0.0096` |
| 7 | **Type_L** | `0.0320` | `0.89%` | `0.0522` | `0.0344` |
| 8 | **Type_M** | `0.0301` | `0.84%` | `0.0325` | `0.0000` |
| 9 | **Type_H** | `0.0117` | `0.33%` | `0.0182` | `0.0039` |

---

## 3. Physical & Engineering Interpretation

1. **Torque `[Nm]` & Rotational Speed `[rpm]` Dominate Mechanical Stress**:
   - **Torque** and **Rotational speed** are the strongest global drivers of failure risk. Physically, mechanical power is proportional to torque multiplied by angular velocity; when torque exceeds 60 Nm at low speeds (or when speed drops below 1380 rpm), the machine enters **Power Failure (`PWF`)** or **Overstrain Failure (`OSF`)** regimes.
2. **Tool Wear `[min]` Drives Cumulative Degradation**:
   - **Tool wear** consistently exhibits a monotonic positive SHAP relationship above 180–200 min: higher tool wear sharply increases log-odds of **Tool Wear Failure (`TWF`)** and amplifies torque-induced **Overstrain Failure (`OSF`)**.
3. **Engineered `Temperature Difference [K]` Captures Heat Dissipation Failures**:
   - The engineered feature (Process Temperature minus Air Temperature) ranks ahead of raw temperatures in Permutation F1 impact because **Heat Dissipation Failure (`HDF`)** occurs specifically when temperature difference drops below 8.6 K combined with rotational speed below 1380 rpm.
