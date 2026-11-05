"""
Member 5 – Part C: Equipment Health Scoring (src/health_scoring.py)

Calculates equipment health scores from ML failure probabilities:
    Health Score = 100 * (1 - Failure Probability)

Health status classification:
    - Good:     80 - 100
    - Moderate: 50 - 79
    - Poor:     Below 50

Returns equipment ID, health score, and health status, and stores results
in the shared database and reports/health_scores.csv.
"""

import math
import logging
import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union

import pandas as pd

try:
    from .database import save_health_score_record
except ImportError:
    from database import save_health_score_record

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
PREDICTION_FILE = PROJECT_DIR / "reports" / "prediction_results.csv"
HEALTH_SCORE_FILE = PROJECT_DIR / "reports" / "health_scores.csv"


def compute_health_score(failure_probability: float) -> Tuple[float, str]:
    """Compute the equipment health score (0-100) and categorical status.

    Formula:
        Health Score = 100 * (1 - Failure Probability)

    Classification:
        - Good:     80 - 100
        - Moderate: 50 - 79
        - Poor:     Below 50
    """
    prob = float(failure_probability)
    if not math.isfinite(prob) or not (0.0 <= prob <= 1.0):
        raise ValueError(f"failure_probability must be between 0 and 1 (got {failure_probability})")

    health_score = round(100.0 * (1.0 - prob), 2)
    if health_score >= 80.0:
        health_status = "Good"
    elif health_score >= 50.0:
        health_status = "Moderate"
    else:
        health_status = "Poor"

    return health_score, health_status


def evaluate_equipment_health(
    equipment_id: str,
    failure_probability: float,
    save_to_db: bool = False,
    timestamp: Optional[Union[datetime.datetime, str]] = None,
) -> Dict[str, Any]:
    """Calculate health score and return equipment_id, health_score, and health_status.

    Optionally persists the result to the shared database.
    """
    eq_id = str(equipment_id).strip()
    if not eq_id:
        raise ValueError("equipment_id cannot be empty")

    score, status = compute_health_score(failure_probability)

    if save_to_db:
        save_health_score_record(
            equipment_id=eq_id,
            failure_probability=float(failure_probability),
            health_score=score,
            health_status=status,
            timestamp=timestamp,
        )

    return {
        "equipment_id": eq_id,
        "failure_probability": round(float(failure_probability), 4),
        "health_score": score,
        "health_status": status,
    }


def save_health_score(
    equipment_id: str,
    failure_probability: float,
    timestamp: Optional[Union[datetime.datetime, str]] = None,
) -> Tuple[float, str]:
    """Calculate the health score, persist it to the database, and return (score, status)."""
    result = evaluate_equipment_health(
        equipment_id=equipment_id,
        failure_probability=failure_probability,
        save_to_db=True,
        timestamp=timestamp,
    )
    return result["health_score"], result["health_status"]


def score_equipment_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Compute health_score and health_status for a DataFrame of predictions."""
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    work = df.copy()
    eq_col = "equipment_id" if "equipment_id" in work.columns else "Equipment ID"
    prob_col = (
        "failure_probability"
        if "failure_probability" in work.columns
        else "Failure Probability"
    )

    if eq_col not in work.columns or prob_col not in work.columns:
        raise ValueError(
            "DataFrame must contain equipment_id ('Equipment ID') and failure_probability ('Failure Probability')."
        )

    # Deduplicate by equipment_id keeping latest record
    work[eq_col] = work[eq_col].astype(str).str.strip()
    work = work.drop_duplicates(subset=[eq_col], keep="last").copy()

    scores = []
    statuses = []
    for prob in work[prob_col].astype(float):
        sc, st = compute_health_score(prob)
        scores.append(sc)
        statuses.append(st)

    result_df = pd.DataFrame(
        {
            "equipment_id": work[eq_col].values,
            "failure_probability": work[prob_col].astype(float).round(4).values,
            "health_score": scores,
            "health_status": statuses,
        }
    )
    return result_df


def main() -> None:
    print("\n==================================================")
    print("MEMBER 5 (PART C): EQUIPMENT HEALTH SCORING")
    print("==================================================")

    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(f"Prediction file not found: {PREDICTION_FILE}")

    preds_df = pd.read_csv(PREDICTION_FILE)
    scored_df = score_equipment_dataframe(preds_df)

    HEALTH_SCORE_FILE.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(HEALTH_SCORE_FILE, index=False)

    print(f"\nTotal unique equipment scored: {len(scored_df)}")
    print(f"Average Health Score: {scored_df['health_score'].mean():.2f} / 100.0")
    print("\nHealth Status Distribution:")
    print(scored_df["health_status"].value_counts().to_string())

    print("\nSample Health Scores (first 10 records):")
    print(scored_df.head(10).to_string(index=False))
    print(f"\nHealth scores saved to: {HEALTH_SCORE_FILE}")


if __name__ == "__main__":
    main()
