import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sn

# Locate project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "equipment_cleaned.csv"
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
plt.close()

print(f"\nChart saved to: {chart_path}")


sensor_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

print("\n" + "=" * 55)
print("Sensor Measurement Analysis")
print("=" * 55)

# Calculate and display average sensor readings by failure status
sensor_means = df.groupby("Machine failure")[sensor_columns].mean()
print("\nAverage Sensor Readings by Operational Status (0: No Failure, 1: Failure):")
print(sensor_means.T.to_string())

# Create subplots for histograms (left) and boxplots (right)
fig, axes = plt.subplots(nrows=len(sensor_columns), ncols=2, figsize=(14, 18))
palette = {0: "#1f77b4", 1: "#d62728"}

for i, col in enumerate(sensor_columns):
    # Left: Sensor distribution histogram with KDE
    sn.histplot(
        data=df,
        x=col,
        kde=True,
        ax=axes[i, 0],
        color="#2ca02c",
        edgecolor="none"
    )
    axes[i, 0].set_title(f"Distribution: {col}", fontweight="bold")
    axes[i, 0].set_ylabel("Count")

    # Right: Comparative boxplot by failure status
    sn.boxplot(
        data=df,
        x="Machine failure",
        y=col,
        ax=axes[i, 1],
        palette=palette,
        hue="Machine failure",
        legend=False
    )
    axes[i, 1].set_title(f"{col} by Machine Failure Status", fontweight="bold")
    axes[i, 1].set_xticks([0, 1])
    axes[i, 1].set_xticklabels(["No Failure (0)", "Failure (1)"])
    axes[i, 1].set_xlabel("Operational Status")

plt.tight_layout()

# Save sensor plots
sensor_chart_path = REPORTS_DIR / "sensor_distributions.png"
plt.savefig(sensor_chart_path, dpi=300)
plt.close()

print(f"\nSensor chart saved to: {sensor_chart_path}")

# ---------------------------------------------------------------------------
# Correlation analysis and statistical summary
# ---------------------------------------------------------------------------

# Compute correlation matrix for sensor measurements and Machine failure
corr_features = df[sensor_columns + ["Machine failure"]].corr()

# Save statistical summary (overall descriptive stats) to CSV
summary_path = REPORTS_DIR / "statistical_summary.csv"
# Use pandas' describe to get summary and write to CSV
df.describe(include="all").to_csv(summary_path)
print(f"\nStatistical summary saved to: {summary_path}")

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sn.heatmap(
    corr_features,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    linecolor="gray",
)
plt.title("Correlation Heatmap of Sensor Measurements and Machine Failure")
plt.tight_layout()
heatmap_path = REPORTS_DIR / "correlation_heatmap.png"
plt.savefig(heatmap_path, dpi=300)
plt.close()

print(f"\nCorrelation heatmap saved to: {heatmap_path}")