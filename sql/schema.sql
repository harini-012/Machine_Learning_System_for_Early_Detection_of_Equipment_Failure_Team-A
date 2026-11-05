-- ============================================================================
-- Milestone 3 - Equipment Predictive Maintenance System
-- Database Name: equipment_maintenance_db
-- Compatible with: Cloud PostgreSQL (Supabase, Neon, Aiven, Railway)
-- ============================================================================

-- 1. Equipment Master Table
CREATE TABLE IF NOT EXISTS equipment (
    equipment_id VARCHAR(64) PRIMARY KEY,
    product_id VARCHAR(64),
    equipment_type CHAR(1) CHECK (equipment_type IN ('L', 'M', 'H')),
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- 2. Sensor Readings Table (Member 1 & Member 2 inputs)
CREATE TABLE IF NOT EXISTS sensor_readings (
    id SERIAL PRIMARY KEY,
    equipment_id VARCHAR(64) NOT NULL REFERENCES equipment(equipment_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    air_temperature DOUBLE PRECISION NOT NULL,
    process_temperature DOUBLE PRECISION NOT NULL,
    rotational_speed DOUBLE PRECISION NOT NULL,
    torque DOUBLE PRECISION NOT NULL,
    tool_wear DOUBLE PRECISION NOT NULL,
    equipment_type CHAR(1) NOT NULL CHECK (equipment_type IN ('L', 'M', 'H'))
);

CREATE INDEX IF NOT EXISTS idx_sensor_readings_eq_ts ON sensor_readings (equipment_id, ts DESC);

-- 3. ML Predictions Table (Member 1 & Member 5 Health Score)
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    equipment_id VARCHAR(64) NOT NULL REFERENCES equipment(equipment_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    failure_predicted BOOLEAN NOT NULL,
    failure_probability DOUBLE PRECISION NOT NULL CHECK (failure_probability >= 0.0 AND failure_probability <= 1.0),
    prediction_label VARCHAR(32) DEFAULT 'No Failure',
    health_score DOUBLE PRECISION,
    health_status VARCHAR(32)
);

CREATE INDEX IF NOT EXISTS idx_predictions_eq_ts ON predictions (equipment_id, ts DESC);

-- 4. Anomaly Detection & Risk Classification Table (Member 2)
CREATE TABLE IF NOT EXISTS anomaly_results (
    id SERIAL PRIMARY KEY,
    equipment_id VARCHAR(64) NOT NULL REFERENCES equipment(equipment_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    anomaly_count INTEGER NOT NULL DEFAULT 0,
    abnormal_sensors TEXT DEFAULT 'None',
    anomalies JSONB DEFAULT '[]'::jsonb,
    risk_level VARCHAR(32) NOT NULL CHECK (risk_level IN ('Healthy', 'Warning', 'Critical')),
    risk_summary TEXT
);

CREATE INDEX IF NOT EXISTS idx_anomaly_results_eq_ts ON anomaly_results (equipment_id, ts DESC);

-- 5. Maintenance Records & Recommendations Table (Member 3)
CREATE TABLE IF NOT EXISTS maintenance_records (
    id SERIAL PRIMARY KEY,
    equipment_id VARCHAR(64) NOT NULL REFERENCES equipment(equipment_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    issue VARCHAR(128) NOT NULL,
    action_type VARCHAR(128) DEFAULT 'Inspection',
    recommendation TEXT NOT NULL,
    risk_level VARCHAR(32) NOT NULL CHECK (risk_level IN ('Healthy', 'Warning', 'Critical')),
    estimated_downtime VARCHAR(64) DEFAULT 'Not specified',
    status VARCHAR(32) DEFAULT 'Pending' CHECK (status IN ('Pending', 'In Progress', 'Completed')),
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_maintenance_records_eq_ts ON maintenance_records (equipment_id, ts DESC);

-- 6. Automated Alerts Table (Member 4)
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    alert_id VARCHAR(64) UNIQUE NOT NULL,
    equipment_id VARCHAR(64) NOT NULL REFERENCES equipment(equipment_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    risk_level VARCHAR(32) NOT NULL CHECK (risk_level IN ('Warning', 'Critical')),
    failure_probability DOUBLE PRECISION NOT NULL CHECK (failure_probability >= 0.0 AND failure_probability <= 1.0),
    condition VARCHAR(255) NOT NULL DEFAULT 'Failure probability risk',
    description TEXT NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'New' CHECK (status IN ('New', 'Acknowledged', 'Resolved'))
);

CREATE INDEX IF NOT EXISTS idx_alerts_eq_status ON alerts (equipment_id, status);

-- 7. Equipment Health Scores Table (Member 5 Part C)
CREATE TABLE IF NOT EXISTS health_scores (
    id SERIAL PRIMARY KEY,
    equipment_id VARCHAR(64) NOT NULL REFERENCES equipment(equipment_id) ON DELETE CASCADE,
    ts TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    failure_probability DOUBLE PRECISION NOT NULL CHECK (failure_probability >= 0.0 AND failure_probability <= 1.0),
    health_score DOUBLE PRECISION NOT NULL CHECK (health_score >= 0.0 AND health_score <= 100.0),
    health_status VARCHAR(32) NOT NULL CHECK (health_status IN ('Good', 'Moderate', 'Poor'))
);

CREATE INDEX IF NOT EXISTS idx_health_scores_eq_ts ON health_scores (equipment_id, ts DESC);
