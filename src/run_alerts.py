
"""
Member 4 Integration:
Member 2 Anomaly Detection
        +
Member 3 Maintenance Recommendations
        +
Member 4 Automated Alert Generation
"""

import pandas as pd
from pathlib import Path

from .alert_management import generate_alerts
from .maintanence_recommendations import (
    process_equipment_recommendations
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

ANOMALY_FILE = PROJECT_DIR / "reports" / "anomaly_results.csv"
ALERT_FILE = PROJECT_DIR / "reports" / "generated_alerts.csv"


def get_sensor_data(row):
    """Convert Member 2 sensor columns into Member 3 input format."""

    return {
        "air_temperature": float(row["Air temperature [K]"]),
        "process_temperature": float(row["Process temperature [K]"]),
        "rotational_speed": float(row["Rotational speed [rpm]"]),
        "torque": float(row["Torque [Nm]"]),
        "tool_wear": float(row["Tool wear [min]"]),
        "equipment_type": str(row["Type"]).strip().upper()
    }


def main():

    print("\n======================================")
    print("MEMBER 3 + MEMBER 4 INTEGRATION")
    print("======================================")

    if not ANOMALY_FILE.exists():
        raise FileNotFoundError(
            f"Anomaly results not found: {ANOMALY_FILE}"
        )

    # Step 1: Load Member 2 results
    df = pd.read_csv(ANOMALY_FILE)

    print("\nMember 2 equipment records loaded:", len(df))

    # Step 2: Generate Member 3 recommendations
    recommendations_by_equipment = {}

    print("\nGenerating maintenance recommendations...")

    for _, row in df.iterrows():

        equipment_id = str(row["Equipment ID"]).strip()

        sensor_data = get_sensor_data(row)

        recommendation = process_equipment_recommendations(
            equipment_id=equipment_id,
            sensor_data=sensor_data,
            risk_level=str(row["Risk Level"]),
            failure_probability=float(row["Failure Probability"])
        )

        recommendations_by_equipment[equipment_id] = recommendation

    print(
        "Maintenance recommendations generated:",
        len(recommendations_by_equipment)
    )

    # Step 3: Load previously saved alerts
    if ALERT_FILE.exists():
        existing_df = pd.read_csv(ALERT_FILE).fillna("")
        existing_alerts = existing_df.to_dict("records")
    else:
        existing_alerts = []

    print("\nPreviously stored alerts:", len(existing_alerts))

    # Step 4: Generate only new, non-duplicate alerts
    print("\nGenerating automated alerts...")

    new_alerts = generate_alerts(
        anomaly_results=df,
        existing_alerts=existing_alerts,
        recommendations_by_equipment=recommendations_by_equipment
    )

    # Preserve old alerts and append new ones
    all_alerts = existing_alerts + new_alerts

    # Step 5: Save complete alert history
    ALERT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    alert_df = pd.DataFrame(all_alerts)

    alert_df.to_csv(
        ALERT_FILE,
        index=False
    )

    # Step 6: Display results
    print("\n======================================")
    print("INTEGRATED ALERT GENERATION SUMMARY")
    print("======================================")

    print("Total equipment:", len(df))

    print(
        "Healthy equipment:",
        (df["Risk Level"] == "Healthy").sum()
    )

    print("New alerts generated:", len(new_alerts))
    print("Previously stored alerts:", len(existing_alerts))
    print("Total alerts stored:", len(all_alerts))

    if not alert_df.empty:

        print("\nAlert distribution:")
        print(alert_df["risk_level"].value_counts())

        print("\nAlert status distribution:")
        print(alert_df["status"].value_counts())

        print("\nMaintenance action distribution:")
        print(alert_df["action_type"].value_counts())

    print("\nGenerated alerts saved to:")
    print(ALERT_FILE)

    print("\nALERT MANAGEMENT COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
