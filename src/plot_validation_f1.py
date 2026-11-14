import matplotlib.pyplot as plt

models = [
    "Logistic Regression",
    "Decision Tree (max_depth=10)",
    "Decision Tree (max_depth=8, min_samples_leaf=4)",
    "Random Forest (n_estimators=100)",
    "Random Forest (n_estimators=100, max_depth=10)",
    "XGBoost (n_estimators=100, max_depth=3)",
    "XGBoost (n_estimators=200, max_depth=3)",
    "XGBoost (n_estimators=200, max_depth=5)",
]

f1_scores = [0.2460, 0.5152, 0.6429, 0.6966, 0.6931, 0.6596, 0.7010, 0.6372]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, f1_scores, color="#4C78A8")
plt.ylim(0, 1)
plt.xlabel("Model configuration")
plt.ylabel("Validation F1 score")
plt.title("Validation F1 scores for candidate models")
plt.xticks(rotation=45, ha='right')
# annotate bars
for bar, score in zip(bars, f1_scores):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{score:.3f}",
             ha='center', va='bottom')
plt.tight_layout()
output_path = "reports/validation_f1.png"
plt.savefig(output_path)
print(f"Saved chart to {output_path}")
