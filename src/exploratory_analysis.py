import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Locate project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "raw" / "ai4i2020.csv"
REPORTS_DIR = PROJECT_DIR / "reports"

# Create reports folder if needed
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Count machine failures and non-failures
failure_counts = df["Machine failure"].value_counts().reindex(
    [0, 1], fill_value=0
)

# Calculate percentages
total = len(df)
failure_percentages = failure_counts / total * 100

print("Failure Distribution Analysis")
print("-----------------------------")
print(f"Total records: {total}")
print(f"Non-failures (0): {failure_counts[0]}")
print(f"Failures (1): {failure_counts[1]}")
print(f"Non-failure percentage: {failure_percentages[0]:.2f}%")
print(f"Failure percentage: {failure_percentages[1]:.2f}%")

# Create bar chart
plt.figure(figsize=(7, 5))
bars = plt.bar(
    ["No Failure (0)", "Failure (1)"],
    failure_counts.values
)

plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure")
plt.ylabel("Number of Records")
plt.bar_label(bars)
plt.tight_layout()

# Save chart
chart_path = REPORTS_DIR / "failure_distribution.png"
plt.savefig(chart_path, dpi=300)
plt.show()

print(f"\nChart saved to: {chart_path}")