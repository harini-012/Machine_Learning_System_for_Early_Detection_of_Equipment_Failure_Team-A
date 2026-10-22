
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "equipment_cleaned.csv"
REPORTS_DIR = PROJECT_DIR / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

sns.set_theme(style="whitegrid")

# Sensor columns
sensor_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# 1. Machine failure distribution
failure_counts = df["Machine failure"].value_counts().sort_index()

print("Machine failure counts:")
print(failure_counts)

print("\nFailure percentage:")
print((df["Machine failure"].value_counts(normalize=True) * 100).round(2))

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Machine failure")
plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "failure_distribution.png", dpi=300)
plt.close()

# 2. Sensor measurement distributions
df[sensor_columns].hist(
    figsize=(12, 8),
    bins=30,
    edgecolor="black"
)

plt.suptitle("Distribution of Equipment Sensor Measurements")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "sensor_distributions.png", dpi=300)
plt.close()

# 3. Compare sensor readings by failure status
mean_comparison = df.groupby("Machine failure")[sensor_columns].mean()

print("\nMean sensor values by failure status:")
print(mean_comparison.round(2))

for column in sensor_columns:
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="Machine failure", y=column)
    plt.title(f"{column} by Machine Failure")
    plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
    plt.tight_layout()

    filename = column.lower().replace(" ", "_")
    filename = filename.replace("[", "").replace("]", "")
    filename = filename.replace("/", "_")

    plt.savefig(
        REPORTS_DIR / f"{filename}_by_failure.png",
        dpi=300
    )
    plt.close()

# 4. Correlation heatmap
numeric_df = df.select_dtypes(include="number")
correlation = numeric_df.corr()

plt.figure(figsize=(13, 9))
sns.heatmap(
    correlation,
    cmap="coolwarm",
    center=0,
    annot=False
)
plt.title("Correlation Heatmap of Numerical Variables")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "correlation_heatmap.png", dpi=300)
plt.close()

# 5. Save statistical summary
df.describe(include="all").to_csv(
    REPORTS_DIR / "statistical_summary.csv"
)

print("\nExploratory analysis completed.")
print("Charts and summary saved in:", REPORTS_DIR)
