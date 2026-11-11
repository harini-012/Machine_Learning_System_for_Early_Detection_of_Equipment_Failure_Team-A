import math

from flask import Flask, abort, render_template, request

from src.prediction_service import create_prediction_service


app = Flask(__name__)
predict_reading = create_prediction_service()

SENSOR_FIELDS = (
    ("air_temperature", "Air temperature [K]", 0.01, "300.50"),
    ("process_temperature", "Process temperature [K]", 0.01, "310.20"),
    ("rotational_speed", "Rotational speed [rpm]", 1, "1320"),
    ("torque", "Torque [Nm]", 0, "62.50"),
    ("tool_wear", "Tool wear [min]", 0, "215"),
)

MONITORING_PAGES = {
    "risk": {
        "title": "Risk classification",
        "nav": "Risk",
        "description": "Review the risk level assigned to equipment readings.",
        "empty": "Risk classifications will appear here when risk analysis is connected.",
    },
    "anomalies": {
        "title": "Anomaly detection",
        "nav": "Anomalies",
        "description": "Review sensor readings flagged as abnormal.",
        "empty": "Flagged sensor readings will appear here when anomaly detection is connected.",
    },
    "alerts": {
        "title": "Equipment alerts",
        "nav": "Alerts",
        "description": "Review equipment that requires attention.",
        "empty": "Equipment alerts will appear here when alert records are connected.",
    },
    "maintenance": {
        "title": "Maintenance recommendations",
        "nav": "Maintenance",
        "description": "Review recommended actions for flagged equipment.",
        "empty": "Recommendations will appear here when maintenance records are connected.",
    },
    "history": {
        "title": "Historical trends",
        "nav": "History",
        "description": "Track changes in sensor readings and predictions over time.",
        "empty": "Trends will appear here when saved equipment readings are connected.",
    },
}


@app.context_processor
def shared_template_data():
    return {"monitoring_pages": MONITORING_PAGES}


def parse_readings(form):
    equipment_id = form.get("equipment_id", "").strip()
    equipment_type = form.get("equipment_type", "")

    if not equipment_id or len(equipment_id) > 100:
        raise ValueError("Enter an equipment ID with 1–100 characters.")

    if equipment_type not in {"L", "M", "H"}:
        raise ValueError("Select equipment type L, M, or H.")

    readings = {
        "equipment_id": equipment_id,
        "Type": equipment_type,
    }

    for field_name, feature_name, minimum, _ in SENSOR_FIELDS:
        try:
            value = float(form.get(field_name, ""))
        except ValueError as exc:
            raise ValueError(f"Enter a number for {feature_name}.") from exc

        if not math.isfinite(value) or value < minimum:
            raise ValueError(
                f"{feature_name} must be a finite number of at least {minimum}."
            )

        readings[feature_name] = value

    return readings


def render_prediction(template_name):
    context = {
        # The original test template expects three values per sensor field.
        "sensor_fields": [field[:3] for field in SENSOR_FIELDS],
        "dashboard_fields": SENSOR_FIELDS,
        "values": {},
        "result": None,
        "error": None,
    }

    if request.method == "GET":
        return render_template(template_name, **context)

    context["values"] = request.form

    try:
        readings = parse_readings(request.form)
    except ValueError as exc:
        context["error"] = str(exc)
        return render_template(template_name, **context), 400

    try:
        context["result"] = predict_reading(readings)
    except Exception:
        app.logger.exception("Prediction failed")
        context["error"] = "Prediction is temporarily unavailable. Please try again."
        return render_template(template_name, **context), 500

    return render_template(template_name, **context)


@app.get("/")
def home():
    return render_template("home.html")


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    return render_prediction("dashboard.html")


@app.route("/predict", methods=["GET", "POST"])
def prediction_page():
    return render_prediction("prediction_test.html")


@app.get("/monitoring/<page>")
def monitoring_page(page):
    details = MONITORING_PAGES.get(page)
    if details is None:
        abort(404)

    return render_template("monitoring.html", page=details, page_key=page)


if __name__ == "__main__":
    app.run()