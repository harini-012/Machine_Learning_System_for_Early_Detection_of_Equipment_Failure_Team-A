import math

from flask import Flask, abort, render_template, request, redirect, url_for
from datetime import datetime, timezone
from src.prediction_service import create_prediction_service

from src.database import (
    get_all_equipment,
    get_equipment_by_id,
    upsert_equipment,
    update_equipment,
    delete_equipment,
    insert_sensor_reading,
    save_prediction,
    retrieve_equipment_history,
)
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

    if not equipment_id or len(equipment_id) > 100:
        raise ValueError("Enter an equipment ID with 1–100 characters.")

    # Fetch registered equipment from the database.
    equipment = get_equipment_by_id(equipment_id)

    if equipment is None:
        raise ValueError("Selected equipment is not registered.")

    # Always use the equipment type stored in the database.
    equipment_type = equipment["equipment_type"]

    readings = {
        "equipment_id": equipment_id,
        "Type": equipment_type,
    }

    for field_name, feature_name, minimum, _ in SENSOR_FIELDS:
        try:
            value = float(form.get(field_name, ""))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Enter a number for {feature_name}.") from exc

        if not math.isfinite(value) or value < minimum:
            raise ValueError(
                f"{feature_name} must be a finite number of at least {minimum}."
            )

        readings[feature_name] = value

    return readings
def render_prediction(template_name):
    context = {
        "sensor_fields": [field[:3] for field in SENSOR_FIELDS],
        "dashboard_fields": SENSOR_FIELDS,
        "equipment": get_all_equipment(),
        "values": {},
        "result": None,
        "error": None,
    }

    if request.method == "GET":
        return render_template(template_name, **context)

    context["values"] = request.form

    try:
        readings = parse_readings(request.form)

        # Generate the model prediction.
        result = predict_reading(readings)

        # Use the same timestamp for both database records.
        observation_time = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )

        equipment = get_equipment_by_id(readings["equipment_id"])

        # Save the five sensor readings as a new observation.
        insert_sensor_reading(
            equipment_id=readings["equipment_id"],
            timestamp=observation_time,
            air_temp=readings["Air temperature [K]"],
            process_temp=readings["Process temperature [K]"],
            rotational_speed=readings["Rotational speed [rpm]"],
            torque=readings["Torque [Nm]"],
            tool_wear=readings["Tool wear [min]"],
            equipment_type=readings["Type"],
            product_id=equipment.get("product_id"),
        )

        # Save the corresponding model prediction.
        save_prediction(
            equipment_id=result["equipment_id"],
            failure_pred=result["predicted_failure"],
            failure_probability=result["failure_probability"],
            timestamp=observation_time,
            prediction_label=result["prediction_label"],
        )

        context["result"] = result

    except ValueError as exc:
        context["error"] = str(exc)
        return render_template(template_name, **context), 400

    except Exception:
        app.logger.exception("Prediction or database saving failed")
        context["error"] = (
            "Prediction could not be completed or saved. Please try again."
        )
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


# ---------------------------------------------------------------------------
# Equipment Management Routes (Milestone 4)
# ---------------------------------------------------------------------------

@app.get("/equipment")
def equipment_list():
    equipment = get_all_equipment()

    return render_template(
        "equipment.html",
        equipment=equipment
    )


@app.route("/equipment/add", methods=["GET", "POST"])
def add_equipment():
    if request.method == "POST":
        equipment_id = request.form.get("equipment_id", "").strip()
        product_id = request.form.get("product_id", "").strip() or None
        equipment_type = request.form.get("equipment_type", "").strip().upper()
        description = request.form.get("description", "").strip() or None

        if not equipment_id or len(equipment_id) > 64:
            return render_template(
                "equipment_form.html",
                mode="Add",
                item=request.form,
                error="Equipment ID must contain between 1 and 64 characters."
            ), 400

        if equipment_type not in ("L", "M", "H"):
            return render_template(
                "equipment_form.html",
                mode="Add",
                item=request.form,
                error="Select a valid equipment type."
            ), 400

        if get_equipment_by_id(equipment_id):
            return render_template(
                "equipment_form.html",
                mode="Add",
                item=request.form,
                error="This equipment ID is already registered."
            ), 400

        try:
            upsert_equipment(
                equipment_id=equipment_id,
                product_id=product_id,
                equipment_type=equipment_type,
                description=description
            )
            return redirect(url_for("equipment_list"))

        except Exception:
            app.logger.exception("Failed to register equipment")
            return render_template(
                "equipment_form.html",
                mode="Add",
                item=request.form,
                error="Unable to register equipment. Please try again."
            ), 500

    return render_template(
        "equipment_form.html",
        mode="Add",
        item={},
        error=None
    )


@app.route("/equipment/edit/<equipment_id>", methods=["GET", "POST"])
def edit_equipment(equipment_id):
    item = get_equipment_by_id(equipment_id)

    if item is None:
        abort(404)

    if request.method == "POST":
        product_id = request.form.get("product_id", "").strip() or None
        equipment_type = request.form.get("equipment_type", "").strip().upper()
        description = request.form.get("description", "").strip() or None

        if equipment_type not in ("L", "M", "H"):
            return render_template(
                "equipment_form.html",
                mode="Edit",
                item={**item, **request.form},
                error="Select a valid equipment type."
            ), 400

        try:
            update_equipment(
                equipment_id=equipment_id,
                product_id=product_id,
                equipment_type=equipment_type,
                description=description
            )
            return redirect(url_for("equipment_list"))

        except Exception:
            app.logger.exception("Failed to update equipment")
            return render_template(
                "equipment_form.html",
                mode="Edit",
                item={**item, **request.form},
                error="Unable to update equipment. Please try again."
            ), 500

    return render_template(
        "equipment_form.html",
        mode="Edit",
        item=item,
        error=None
    )

@app.get("/equipment/<equipment_id>")
def equipment_detail(equipment_id):
    history = retrieve_equipment_history(equipment_id)

    if not history["equipment"]:
        abort(404)

    equipment = history["equipment"][0]

    return render_template(
        "equipment_detail.html",
        equipment=equipment,
        sensor_readings=history["sensor_readings"],
        predictions=history["predictions"],
        anomalies=history["anomalies"],
        maintenance=history["maintenance"],
        alerts=history["alerts"],
        health_scores=history["health_scores"],
    )
@app.post("/equipment/delete/<equipment_id>")
def remove_equipment(equipment_id):
    item = get_equipment_by_id(equipment_id)

    if item is None:
        abort(404)

    try:
        delete_equipment(equipment_id)
        return redirect(url_for("equipment_list"))

    except Exception:
        app.logger.exception("Failed to delete equipment")
        abort(500)

if __name__ == "__main__":
    app.run()