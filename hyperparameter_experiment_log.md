# Hyperparameter Experiment Log

This log records the validation performance of various candidate models evaluated by `src/tune_model.py` and the final test performance of the selected best model.

## Validation Results

| Model | Configuration | Validation F1 | Validation Recall | Validation ROC-AUC |
| --- | --- | ---: | ---: | ---: |
| Logistic Regression | C=1.0, class_weight=balanced | 0.2460 | 0.8519 | 0.8926 |
| Decision Tree | max_depth=10, class_weight=balanced | 0.5152 | 0.6296 | 0.8062 |
| Decision Tree | max_depth=8, min_samples_leaf=4, unweighted | 0.6429 | 0.5000 | 0.8877 |
| Random Forest | n_estimators=100, max_depth=None, unweighted | 0.6966 | 0.5741 | 0.9586 |
| Random Forest | n_estimators=100, max_depth=10, class_weight=balanced | 0.6931 | 0.6481 | 0.9742 |
| XGBoost | n_estimators=100, max_depth=3, lr=0.1, scale_pos_weight=1.0 | 0.6596 | 0.5741 | 0.9812 |
| XGBoost | n_estimators=200, max_depth=3, lr=0.1, scale_pos_weight=1.0 | **0.7010** (best) | 0.6296 | 0.9776 |
| XGBoost | n_estimators=200, max_depth=5, lr=0.1, scale_pos_weight=balanced | 0.6372 | 0.6667 | 0.9756 |

## Selected Best Model (XGBoost, n_estimators=200, max_depth=3, lr=0.1, scale_pos_weight=1.0)

| Metric | Test Set |
| --- | ---: |
| Accuracy | 0.9900 |
| Precision | 0.9444 |
| Recall | 0.7500 |
| F1-Score | 0.8361 |
| ROC-AUC | 0.9808 |

The best model was saved to `models/best_model.joblib`.
