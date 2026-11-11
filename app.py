import math

from flask import Flask, redirect, render_template, request, url_for

from src.prediction_service import create_prediction_service


app = Flask(__name__)
predict_reading = create_prediction_service()

SENSOR_FIELDS = (
    ("air_temperature", "Air temperature [K]", 0.01),
    ("process_temperature", "Process temperature [K]", 0.01),
    ("rotational_speed", "Rotational speed [rpm]", 1),
    ("torque", "Torque [Nm]", 0),
    ("tool_wear", "Tool wear [min]", 0),
)


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

    for field_name, feature_name, minimum in SENSOR_FIELDS:
        try:
            value = float(form.get(field_name, ""))
        except ValueError as exc:
            raise ValueError(
                f"Enter a number for {feature_name}."
            ) from exc

        if not math.isfinite(value) or value < minimum:
            raise ValueError(
                f"{feature_name} must be at least {minimum}."
            )

        readings[feature_name] = value

    return readings


@app.get("/")
def home():
    return redirect(url_for("dashboard"))


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    context = {
        "sensor_fields": SENSOR_FIELDS,
        "values": {},
        "result": None,
        "error": None,
    }

    if request.method == "GET":
        return render_template("dashboard.html", **context)

    context["values"] = request.form

    try:
        readings = parse_readings(request.form)
    except ValueError as exc:
        context["error"] = str(exc)
        return render_template("dashboard.html", **context), 400

    try:
        result = predict_reading(readings)
        context["result"] = result
    except Exception:
        app.logger.exception("Prediction failed")
        context["error"] = (
            "Prediction failed. Check the Flask terminal for details."
        )
        return render_template("dashboard.html", **context), 500

    return render_template("dashboard.html", **context)


@app.route("/predict", methods=["GET", "POST"])
def prediction_page():
    context = {
        "sensor_fields": SENSOR_FIELDS,
        "values": {},
        "result": None,
        "error": None,
    }

    if request.method == "GET":
        return render_template("prediction_test.html", **context)

    context["values"] = request.form

    try:
        readings = parse_readings(request.form)
    except ValueError as exc:
        context["error"] = str(exc)
        return render_template("prediction_test.html", **context), 400

    try:
        context["result"] = predict_reading(readings)
    except Exception:
        app.logger.exception("Prediction failed")
        context["error"] = (
            "Prediction failed. Check the Flask terminal for details."
        )
        return render_template("prediction_test.html", **context), 500

    return render_template("prediction_test.html", **context)


if __name__ == "__main__":
    app.run(debug=True)