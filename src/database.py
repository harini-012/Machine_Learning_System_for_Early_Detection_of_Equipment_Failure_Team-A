"""
Member 5 – Shared Database Connection Module (src/database.py)

Provides reusable database connection and CRUD functions for:
- Connecting to Cloud PostgreSQL (Supabase, Neon, Aiven, Railway) via DATABASE_URL,
  with automatic zero-config fallback to a local SQLite database
  (data/equipment_maintenance_db.sqlite) so all team members can run and test
  the full pipeline offline or in the cloud using the exact same table structure.
- Creating and initializing the 5 required Milestone 3 tables (plus anomaly_results
  and health_scores):
    1. equipment
    2. sensor_readings
    3. predictions
    4. alerts
    5. maintenance_records
    (+ anomaly_results, health_scores)
- Inserting sensor readings (Member 1 & 2)
- Saving ML predictions (Member 1)
- Saving anomaly and risk results (Member 2)
- Saving maintenance recommendations (Member 3)
- Saving, updating, and retrieving alerts (Member 4)
- Saving equipment health scores and retrieving equipment history (Member 5)
"""

import os
import json
import uuid
import sqlite3
import logging
import datetime
from pathlib import Path
from contextlib import contextmanager
from typing import List, Dict, Any, Optional, Union

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, execute_batch
    HAS_PSYCOPG2 = True
except ImportError:
    psycopg2 = None
    RealDictCursor = None
    execute_batch = None
    HAS_PSYCOPG2 = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SQLITE_PATH = PROJECT_DIR / "data" / "equipment_maintenance_db.sqlite"
SCHEMA_SQL_PATH = PROJECT_DIR / "sql" / "schema.sql"


def _load_dotenv_if_present() -> None:
    """Load key=value pairs from .env in project root if not already set."""
    env_file = PROJECT_DIR / ".env"
    if not env_file.exists():
        return
    try:
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception as exc:
        logger.debug("Could not parse .env file: %s", exc)


_load_dotenv_if_present()


def is_postgres_configured() -> bool:
    """Return True if DATABASE_URL points to a PostgreSQL instance."""
    db_url = os.getenv("DATABASE_URL", "").strip()
    return db_url.startswith("postgres://") or db_url.startswith("postgresql://")


def get_database_backend() -> str:
    """Return 'postgresql' if Cloud PostgreSQL is configured, else 'sqlite'."""
    return "postgresql" if is_postgres_configured() else "sqlite"


def get_sqlite_path() -> Path:
    """Return the path to the SQLite database file."""
    db_url = os.getenv("DATABASE_URL", "").strip()
    if db_url.startswith("sqlite:///"):
        custom_path = db_url.replace("sqlite:///", "", 1)
        return Path(custom_path)
    return DEFAULT_SQLITE_PATH


def get_connection():
    """Create and return a new database connection (PostgreSQL or SQLite fallback).

    If DATABASE_URL is set to a PostgreSQL URI (postgres:// or postgresql://),
    connects to Cloud PostgreSQL using psycopg2.
    Otherwise connects to the shared SQLite database file at
    data/equipment_maintenance_db.sqlite.
    """
    _load_dotenv_if_present()
    db_url = os.getenv("DATABASE_URL", "").strip()

    if is_postgres_configured():
        if not HAS_PSYCOPG2:
            raise RuntimeError(
                "DATABASE_URL is set to PostgreSQL, but psycopg2 is not installed. "
                "Install it with: pip install psycopg2-binary"
            )
        try:
            conn = psycopg2.connect(db_url, cursor_factory=RealDictCursor)
            return conn
        except Exception as exc:
            logger.exception("Failed to connect to Cloud PostgreSQL: %s", exc)
            raise

    sqlite_path = get_sqlite_path()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@contextmanager
def managed_connection():
    """Context manager that commits on success, rolls back on error, and always closes."""
    conn = get_connection()
    backend = get_database_backend()
    try:
        yield conn, backend
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error("Database operation failed (%s): %s", backend, exc)
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _normalize_timestamp(ts: Optional[Union[datetime.datetime, str]] = None) -> str:
    """Convert a datetime or ISO string to an ISO-8601 UTC timestamp string."""
    if ts is None:
        return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    if isinstance(ts, datetime.datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)
        return ts.isoformat(timespec="seconds")
    return str(ts)


def _rows_to_dicts(rows: List[Any]) -> List[Dict[str, Any]]:
    """Convert sqlite3.Row or RealDictRow objects into plain Python dicts."""
    result = []
    for row in rows:
        d = dict(row)
        # Parse JSON strings in anomalies column if stored as text in SQLite
        if "anomalies" in d and isinstance(d["anomalies"], str):
            try:
                d["anomalies"] = json.loads(d["anomalies"])
            except Exception:
                pass
        result.append(d)
    return result


# ---------------------------------------------------------------------------
# Database Schema Initialization (PostgreSQL & SQLite)
# ---------------------------------------------------------------------------

SQLITE_SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS equipment (
        equipment_id TEXT PRIMARY KEY,
        product_id TEXT,
        equipment_type TEXT CHECK (equipment_type IN ('L', 'M', 'H')),
        description TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS sensor_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        equipment_id TEXT NOT NULL REFERENCES equipment(equipment_id) ON DELETE CASCADE,
        ts TEXT NOT NULL,
        air_temperature REAL NOT NULL,
        process_temperature REAL NOT NULL,
        rotational_speed REAL NOT NULL,
        torque REAL NOT NULL,
        tool_wear REAL NOT NULL,
        equipment_type TEXT NOT NULL CHECK (equipment_type IN ('L', 'M', 'H'))
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_sensor_readings_eq_ts ON sensor_readings (equipment_id, ts DESC);",
    """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        equipment_id TEXT NOT NULL REFERENCES equipment(equipment_id) ON DELETE CASCADE,
        ts TEXT NOT NULL,
        failure_predicted INTEGER NOT NULL,
        failure_probability REAL NOT NULL CHECK (failure_probability >= 0.0 AND failure_probability <= 1.0),
        prediction_label TEXT DEFAULT 'No Failure',
        health_score REAL,
        health_status TEXT
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_predictions_eq_ts ON predictions (equipment_id, ts DESC);",
    """
    CREATE TABLE IF NOT EXISTS anomaly_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        equipment_id TEXT NOT NULL REFERENCES equipment(equipment_id) ON DELETE CASCADE,
        ts TEXT NOT NULL,
        anomaly_count INTEGER NOT NULL DEFAULT 0,
        abnormal_sensors TEXT DEFAULT 'None',
        anomalies TEXT DEFAULT '[]',
        risk_level TEXT NOT NULL CHECK (risk_level IN ('Healthy', 'Warning', 'Critical')),
        risk_summary TEXT
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_anomaly_results_eq_ts ON anomaly_results (equipment_id, ts DESC);",
    """
    CREATE TABLE IF NOT EXISTS maintenance_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        equipment_id TEXT NOT NULL REFERENCES equipment(equipment_id) ON DELETE CASCADE,
        ts TEXT NOT NULL,
        issue TEXT NOT NULL,
        action_type TEXT DEFAULT 'Inspection',
        recommendation TEXT NOT NULL,
        risk_level TEXT NOT NULL CHECK (risk_level IN ('Healthy', 'Warning', 'Critical')),
        estimated_downtime TEXT DEFAULT 'Not specified',
        status TEXT DEFAULT 'Pending' CHECK (status IN ('Pending', 'In Progress', 'Completed')),
        completed_at TEXT
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_maintenance_records_eq_ts ON maintenance_records (equipment_id, ts DESC);",
    """
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_id TEXT UNIQUE NOT NULL,
        equipment_id TEXT NOT NULL REFERENCES equipment(equipment_id) ON DELETE CASCADE,
        ts TEXT NOT NULL,
        risk_level TEXT NOT NULL CHECK (risk_level IN ('Warning', 'Critical')),
        failure_probability REAL NOT NULL CHECK (failure_probability >= 0.0 AND failure_probability <= 1.0),
        condition TEXT NOT NULL DEFAULT 'Failure probability risk',
        description TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'New' CHECK (status IN ('New', 'Acknowledged', 'Resolved'))
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_alerts_eq_status ON alerts (equipment_id, status);",
    """
    CREATE TABLE IF NOT EXISTS health_scores (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        equipment_id TEXT NOT NULL REFERENCES equipment(equipment_id) ON DELETE CASCADE,
        ts TEXT NOT NULL,
        failure_probability REAL NOT NULL CHECK (failure_probability >= 0.0 AND failure_probability <= 1.0),
        health_score REAL NOT NULL CHECK (health_score >= 0.0 AND health_score <= 100.0),
        health_status TEXT NOT NULL CHECK (health_status IN ('Good', 'Moderate', 'Poor'))
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_health_scores_eq_ts ON health_scores (equipment_id, ts DESC);",
]


def init_database() -> None:
    """Create all required tables and indexes if they do not exist."""
    with managed_connection() as (conn, backend):
        if backend == "postgresql":
            schema_sql = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
            with conn.cursor() as cur:
                cur.execute(schema_sql)
        else:
            cur = conn.cursor()
            for stmt in SQLITE_SCHEMA_STATEMENTS:
                cur.execute(stmt)


def _ensure_equipment_in_tx(
    conn,
    backend: str,
    equipment_id: str,
    product_id: Optional[str] = None,
    equipment_type: Optional[str] = "L",
    description: Optional[str] = None,
) -> None:
    """Ensure the equipment record exists within an active transaction."""
    eq_type = (str(equipment_type).strip().upper() if equipment_type else "L")
    if eq_type not in ("L", "M", "H"):
        eq_type = "L"
    desc = description or f"Industrial Equipment {equipment_id} (Type {eq_type})"

    if backend == "postgresql":
        sql = """
        INSERT INTO equipment (equipment_id, product_id, equipment_type, description)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (equipment_id) DO UPDATE SET
            product_id = COALESCE(EXCLUDED.product_id, equipment.product_id),
            equipment_type = COALESCE(EXCLUDED.equipment_type, equipment.equipment_type),
            updated_at = CURRENT_TIMESTAMP;
        """
        with conn.cursor() as cur:
            cur.execute(sql, (equipment_id, product_id, eq_type, desc))
    else:
        sql = """
        INSERT INTO equipment (equipment_id, product_id, equipment_type, description)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(equipment_id) DO UPDATE SET
            product_id = COALESCE(excluded.product_id, equipment.product_id),
            equipment_type = COALESCE(excluded.equipment_type, equipment.equipment_type),
            updated_at = CURRENT_TIMESTAMP;
        """
        conn.execute(sql, (equipment_id, product_id, eq_type, desc))


def upsert_equipment(
    equipment_id: str,
    product_id: Optional[str] = None,
    equipment_type: str = "L",
    description: Optional[str] = None,
) -> None:
    """Insert or update an equipment record in the equipment table."""
    init_database()
    with managed_connection() as (conn, backend):
        _ensure_equipment_in_tx(
            conn,
            backend,
            equipment_id=str(equipment_id).strip(),
            product_id=product_id,
            equipment_type=equipment_type,
            description=description,
        )


# ---------------------------------------------------------------------------
# 1. Sensor Readings (Member 1 & Member 2)
# ---------------------------------------------------------------------------

def insert_sensor_reading(
    equipment_id: str,
    timestamp: Optional[Union[datetime.datetime, str]] = None,
    air_temp: float = 300.0,
    process_temp: float = 310.0,
    rotational_speed: float = 1500.0,
    torque: float = 40.0,
    tool_wear: float = 0.0,
    equipment_type: str = "L",
    product_id: Optional[str] = None,
) -> None:
    """Insert a single sensor reading row into sensor_readings."""
    init_database()
    eq_id = str(equipment_id).strip()
    eq_type = str(equipment_type).strip().upper()
    if eq_type not in ("L", "M", "H"):
        eq_type = "L"
    ts_str = _normalize_timestamp(timestamp)

    with managed_connection() as (conn, backend):
        _ensure_equipment_in_tx(conn, backend, eq_id, product_id=product_id, equipment_type=eq_type)
        params = (
            eq_id,
            ts_str,
            float(air_temp),
            float(process_temp),
            float(rotational_speed),
            float(torque),
            float(tool_wear),
            eq_type,
        )
        if backend == "postgresql":
            sql = """
            INSERT INTO sensor_readings (
                equipment_id, ts, air_temperature, process_temperature,
                rotational_speed, torque, tool_wear, equipment_type
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """
            with conn.cursor() as cur:
                cur.execute(sql, params)
        else:
            sql = """
            INSERT INTO sensor_readings (
                equipment_id, ts, air_temperature, process_temperature,
                rotational_speed, torque, tool_wear, equipment_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """
            conn.execute(sql, params)


# ---------------------------------------------------------------------------
# 2. ML Predictions (Member 1 & Member 5 Health Score)
# ---------------------------------------------------------------------------

def save_prediction(
    equipment_id: str,
    failure_pred: Union[bool, int],
    failure_probability: float,
    timestamp: Optional[Union[datetime.datetime, str]] = None,
    prediction_label: Optional[str] = None,
    health_score: Optional[float] = None,
    health_status: Optional[str] = None,
) -> None:
    """Save the ML model's failure prediction and automatically compute health score."""
    init_database()
    eq_id = str(equipment_id).strip()
    pred_bool = bool(int(failure_pred))
    prob = float(failure_probability)
    label = prediction_label or ("Failure" if pred_bool else "No Failure")
    ts_str = _normalize_timestamp(timestamp)

    if health_score is None or health_status is None:
        health_score = round(100.0 * (1.0 - prob), 2)
        if health_score >= 80.0:
            health_status = "Good"
        elif health_score >= 50.0:
            health_status = "Moderate"
        else:
            health_status = "Poor"

    with managed_connection() as (conn, backend):
        _ensure_equipment_in_tx(conn, backend, eq_id)
        if backend == "postgresql":
            sql = """
            INSERT INTO predictions (
                equipment_id, ts, failure_predicted, failure_probability,
                prediction_label, health_score, health_status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s);
            """
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (eq_id, ts_str, pred_bool, prob, label, float(health_score), health_status),
                )
        else:
            sql = """
            INSERT INTO predictions (
                equipment_id, ts, failure_predicted, failure_probability,
                prediction_label, health_score, health_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """
            conn.execute(
                sql,
                (eq_id, ts_str, int(pred_bool), prob, label, float(health_score), health_status),
            )


# ---------------------------------------------------------------------------
# 3. Anomaly Detection & Risk Classification (Member 2)
# ---------------------------------------------------------------------------

def save_anomaly_and_risk(
    equipment_id: str,
    anomalies: Union[List[Dict[str, Any]], str],
    risk_level: str,
    timestamp: Optional[Union[datetime.datetime, str]] = None,
    anomaly_count: Optional[int] = None,
    abnormal_sensors: Optional[str] = None,
    risk_summary: Optional[str] = None,
) -> None:
    """Store anomaly detection results and overall equipment risk level."""
    init_database()
    eq_id = str(equipment_id).strip()
    risk = str(risk_level).strip().capitalize()
    if risk not in ("Healthy", "Warning", "Critical"):
        risk = "Healthy"
    ts_str = _normalize_timestamp(timestamp)

    if isinstance(anomalies, str):
        abnormal_str = abnormal_sensors or anomalies
        sensor_list = (
            [s.strip() for s in anomalies.split(",") if s.strip() and s.strip().lower() != "none"]
            if anomalies
            else []
        )
        anomalies_obj = [{"sensor": s} for s in sensor_list]
    elif isinstance(anomalies, list):
        anomalies_obj = anomalies
        if abnormal_sensors is None:
            names = [
                str(a.get("sensor", a)) if isinstance(a, dict) else str(a)
                for a in anomalies_obj
            ]
            abnormal_str = ", ".join(names) if names else "None"
        else:
            abnormal_str = abnormal_sensors
    else:
        anomalies_obj = []
        abnormal_str = abnormal_sensors or "None"

    count = anomaly_count if anomaly_count is not None else len(anomalies_obj)
    summary = risk_summary or f"{risk} equipment risk. Abnormal sensors: {abnormal_str}"
    anomalies_json = json.dumps(anomalies_obj)

    with managed_connection() as (conn, backend):
        _ensure_equipment_in_tx(conn, backend, eq_id)
        if backend == "postgresql":
            sql = """
            INSERT INTO anomaly_results (
                equipment_id, ts, anomaly_count, abnormal_sensors,
                anomalies, risk_level, risk_summary
            ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s);
            """
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (eq_id, ts_str, int(count), abnormal_str, anomalies_json, risk, summary),
                )
        else:
            sql = """
            INSERT INTO anomaly_results (
                equipment_id, ts, anomaly_count, abnormal_sensors,
                anomalies, risk_level, risk_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """
            conn.execute(
                sql,
                (eq_id, ts_str, int(count), abnormal_str, anomalies_json, risk, summary),
            )


# ---------------------------------------------------------------------------
# 4. Maintenance Recommendations (Member 3)
# ---------------------------------------------------------------------------

def save_maintenance_recommendation(
    equipment_id: str,
    issue: Optional[str] = None,
    recommendation: Optional[str] = None,
    risk_level: str = "Healthy",
    timestamp: Optional[Union[datetime.datetime, str]] = None,
    *,
    primary_issue: Optional[str] = None,
    action_type: str = "Inspection",
    recommendations: Optional[Union[str, List[str]]] = None,
    estimated_downtime: str = "Not specified",
    status: str = "Pending",
) -> None:
    """Persist a maintenance recommendation in maintenance_records.

    Supports both:
    - Positional/standard call:
        save_maintenance_recommendation(equipment_id, issue, recommendation, risk_level)
    - Member 3's keyword call in src/maintanence_recommendations.py:
        save_maintenance_recommendation(
            equipment_id=..., primary_issue=..., action_type=...,
            recommendations=..., risk_level=...
        )
    """
    init_database()
    eq_id = str(equipment_id).strip()
    resolved_issue = primary_issue or issue or "General Inspection"
    if isinstance(recommendations, list):
        resolved_rec = "; ".join(str(r) for r in recommendations)
    else:
        resolved_rec = recommendations or recommendation or "Perform standard inspection."

    risk = str(risk_level).strip().capitalize()
    if risk not in ("Healthy", "Warning", "Critical"):
        risk = "Healthy"
    ts_str = _normalize_timestamp(timestamp)

    with managed_connection() as (conn, backend):
        _ensure_equipment_in_tx(conn, backend, eq_id)
        params = (
            eq_id,
            ts_str,
            str(resolved_issue),
            str(action_type),
            str(resolved_rec),
            risk,
            str(estimated_downtime),
            str(status),
        )
        if backend == "postgresql":
            sql = """
            INSERT INTO maintenance_records (
                equipment_id, ts, issue, action_type, recommendation,
                risk_level, estimated_downtime, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """
            with conn.cursor() as cur:
                cur.execute(sql, params)
        else:
            sql = """
            INSERT INTO maintenance_records (
                equipment_id, ts, issue, action_type, recommendation,
                risk_level, estimated_downtime, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """
            conn.execute(sql, params)


# ---------------------------------------------------------------------------
# 5. Automated Alerts (Member 4)
# ---------------------------------------------------------------------------

def save_alert(
    equipment_id: str,
    risk_level: str,
    failure_probability: float,
    description: str,
    status: str = "New",
    timestamp: Optional[Union[datetime.datetime, str]] = None,
    alert_id: Optional[str] = None,
    condition: str = "Failure probability risk",
) -> Optional[str]:
    """Insert or update an alert record in the alerts table.

    Prevents duplicate active alerts (status in ('New', 'Acknowledged')) for the
    same equipment_id and condition. Returns the alert_id saved.
    """
    init_database()
    eq_id = str(equipment_id).strip()
    risk = str(risk_level).strip().capitalize()
    if risk == "Healthy":
        return None
    if risk not in ("Warning", "Critical"):
        raise ValueError(f"Invalid alert risk_level: {risk_level}")

    if status not in ("New", "Acknowledged", "Resolved"):
        raise ValueError(f"Invalid alert status: {status}")

    prob = float(failure_probability)
    cond = str(condition).strip() or "Failure probability risk"
    ts_str = _normalize_timestamp(timestamp)
    aid = str(alert_id).strip() if alert_id else str(uuid.uuid4())

    with managed_connection() as (conn, backend):
        _ensure_equipment_in_tx(conn, backend, eq_id)
        if backend == "postgresql":
            with conn.cursor() as cur:
                # Check if an active alert already exists for this equipment & condition
                cur.execute(
                    """
                    SELECT alert_id FROM alerts
                    WHERE equipment_id = %s
                      AND LOWER(condition) = LOWER(%s)
                      AND status IN ('New', 'Acknowledged')
                    LIMIT 1;
                    """,
                    (eq_id, cond),
                )
                existing = cur.fetchone()
                if existing:
                    existing_id = existing["alert_id"]
                    cur.execute(
                        """
                        UPDATE alerts
                        SET ts = %s,
                            risk_level = %s,
                            failure_probability = %s,
                            description = %s
                        WHERE alert_id = %s;
                        """,
                        (ts_str, risk, prob, description, existing_id),
                    )
                    return existing_id

                cur.execute(
                    """
                    INSERT INTO alerts (
                        alert_id, equipment_id, ts, risk_level,
                        failure_probability, condition, description, status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    (aid, eq_id, ts_str, risk, prob, cond, description, status),
                )
        else:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT alert_id FROM alerts
                WHERE equipment_id = ?
                  AND LOWER(condition) = LOWER(?)
                  AND status IN ('New', 'Acknowledged')
                LIMIT 1;
                """,
                (eq_id, cond),
            )
            existing = cur.fetchone()
            if existing:
                existing_id = existing["alert_id"]
                cur.execute(
                    """
                    UPDATE alerts
                    SET ts = ?,
                        risk_level = ?,
                        failure_probability = ?,
                        description = ?
                    WHERE alert_id = ?;
                    """,
                    (ts_str, risk, prob, description, existing_id),
                )
                return existing_id

            cur.execute(
                """
                INSERT INTO alerts (
                    alert_id, equipment_id, ts, risk_level,
                    failure_probability, condition, description, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (aid, eq_id, ts_str, risk, prob, cond, description, status),
            )
    return aid


def update_alert_status_in_db(alert_id: str, new_status: str) -> Dict[str, Any]:
    """Update an alert's status ('New', 'Acknowledged', 'Resolved') in the database."""
    init_database()
    if new_status not in ("New", "Acknowledged", "Resolved"):
        raise ValueError(f"Invalid status: {new_status}")

    with managed_connection() as (conn, backend):
        if backend == "postgresql":
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM alerts WHERE alert_id = %s;", (alert_id,))
                row = cur.fetchone()
                if not row:
                    raise ValueError(f"Alert ID not found: {alert_id}")
                if row["status"] == "Resolved":
                    raise ValueError("Resolved alerts cannot be reopened.")
                cur.execute(
                    "UPDATE alerts SET status = %s WHERE alert_id = %s;",
                    (new_status, alert_id),
                )
                cur.execute("SELECT * FROM alerts WHERE alert_id = %s;", (alert_id,))
                return dict(cur.fetchone())
        else:
            cur = conn.cursor()
            cur.execute("SELECT * FROM alerts WHERE alert_id = ?;", (alert_id,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Alert ID not found: {alert_id}")
            if row["status"] == "Resolved":
                raise ValueError("Resolved alerts cannot be reopened.")
            cur.execute(
                "UPDATE alerts SET status = ? WHERE alert_id = ?;",
                (new_status, alert_id),
            )
            cur.execute("SELECT * FROM alerts WHERE alert_id = ?;", (alert_id,))
            return dict(cur.fetchone())


def get_alerts(
    equipment_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Retrieve alerts from the database, optionally filtered by equipment_id and/or status."""
    init_database()
    with managed_connection() as (conn, backend):
        placeholder = "%s" if backend == "postgresql" else "?"
        sql = "SELECT * FROM alerts"
        conditions = []
        params: List[Any] = []
        if equipment_id:
            conditions.append(f"equipment_id = {placeholder}")
            params.append(str(equipment_id).strip())
        if status:
            conditions.append(f"status = {placeholder}")
            params.append(str(status).strip())
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY ts DESC"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        sql += ";"

        if backend == "postgresql":
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                return _rows_to_dicts(cur.fetchall())
        else:
            cur = conn.cursor()
            cur.execute(sql, tuple(params))
            return _rows_to_dicts(cur.fetchall())


# ---------------------------------------------------------------------------
# 6. Equipment Health Scoring Persistence (Member 5 Part C)
# ---------------------------------------------------------------------------

def save_health_score_record(
    equipment_id: str,
    failure_probability: float,
    health_score: float,
    health_status: str,
    timestamp: Optional[Union[datetime.datetime, str]] = None,
) -> None:
    """Persist an equipment health score record to health_scores."""
    init_database()
    eq_id = str(equipment_id).strip()
    ts_str = _normalize_timestamp(timestamp)
    with managed_connection() as (conn, backend):
        _ensure_equipment_in_tx(conn, backend, eq_id)
        params = (eq_id, ts_str, float(failure_probability), float(health_score), str(health_status))
        if backend == "postgresql":
            sql = """
            INSERT INTO health_scores (
                equipment_id, ts, failure_probability, health_score, health_status
            ) VALUES (%s, %s, %s, %s, %s);
            """
            with conn.cursor() as cur:
                cur.execute(sql, params)
        else:
            sql = """
            INSERT INTO health_scores (
                equipment_id, ts, failure_probability, health_score, health_status
            ) VALUES (?, ?, ?, ?, ?);
            """
            conn.execute(sql, params)


# ---------------------------------------------------------------------------
# 7. Equipment History & Dashboard Queries (Member 5 Part B & Part D)
# ---------------------------------------------------------------------------

def retrieve_equipment_history(equipment_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """Collect all historical data for a given equipment ID across all tables.

    Returns a dictionary with keys:
    - equipment
    - sensor_readings
    - predictions
    - anomalies
    - maintenance
    - alerts
    - health_scores
    """
    init_database()
    eq_id = str(equipment_id).strip()
    with managed_connection() as (conn, backend):
        placeholder = "%s" if backend == "postgresql" else "?"
        cur = conn.cursor()

        cur.execute(f"SELECT * FROM equipment WHERE equipment_id = {placeholder};", (eq_id,))
        eq_rows = _rows_to_dicts(cur.fetchall())

        cur.execute(
            f"SELECT * FROM sensor_readings WHERE equipment_id = {placeholder} ORDER BY ts ASC, id ASC;",
            (eq_id,),
        )
        sensor = _rows_to_dicts(cur.fetchall())

        cur.execute(
            f"SELECT * FROM predictions WHERE equipment_id = {placeholder} ORDER BY ts ASC, id ASC;",
            (eq_id,),
        )
        preds = _rows_to_dicts(cur.fetchall())

        cur.execute(
            f"SELECT * FROM anomaly_results WHERE equipment_id = {placeholder} ORDER BY ts ASC, id ASC;",
            (eq_id,),
        )
        anomalies = _rows_to_dicts(cur.fetchall())

        cur.execute(
            f"SELECT * FROM maintenance_records WHERE equipment_id = {placeholder} ORDER BY ts ASC, id ASC;",
            (eq_id,),
        )
        maint = _rows_to_dicts(cur.fetchall())

        cur.execute(
            f"SELECT * FROM alerts WHERE equipment_id = {placeholder} ORDER BY ts ASC, id ASC;",
            (eq_id,),
        )
        alerts = _rows_to_dicts(cur.fetchall())

        cur.execute(
            f"SELECT * FROM health_scores WHERE equipment_id = {placeholder} ORDER BY ts ASC, id ASC;",
            (eq_id,),
        )
        scores = _rows_to_dicts(cur.fetchall())

    return {
        "equipment": eq_rows,
        "sensor_readings": sensor,
        "predictions": preds,
        "anomalies": anomalies,
        "maintenance": maint,
        "alerts": alerts,
        "health_scores": scores,
    }


def get_table_counts() -> Dict[str, int]:
    """Return total row counts for each table in the database."""
    init_database()
    tables = [
        "equipment",
        "sensor_readings",
        "predictions",
        "anomaly_results",
        "maintenance_records",
        "alerts",
        "health_scores",
    ]
    counts: Dict[str, int] = {}
    with managed_connection() as (conn, backend):
        cur = conn.cursor()
        for table in tables:
            cur.execute(f"SELECT COUNT(*) AS cnt FROM {table};")
            row = cur.fetchone()
            counts[table] = int(row["cnt"] if isinstance(row, dict) or hasattr(row, "keys") else row[0])
    return counts
