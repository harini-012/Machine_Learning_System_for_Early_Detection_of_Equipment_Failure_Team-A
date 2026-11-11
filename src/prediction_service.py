import xgboost as xgb

from src.model_integration import (
    load_best_model,
    load_feature_columns,
    prepare_features,
)


def create_prediction_service():
    model = load_best_model()
    feature_columns = load_feature_columns()

    def predict_reading(readings):
        features, equipment_ids = prepare_features(readings, feature_columns)

        booster = model.get_booster()
        expected_names = booster.feature_names or features.columns.tolist()

        missing_names = [
            name for name in expected_names if name not in features.columns
        ]
        if missing_names:
            raise ValueError(f"Missing model features: {', '.join(missing_names)}")

        ordered_features = features.loc[:, expected_names]
        model_input = xgb.DMatrix(
            ordered_features.to_numpy(dtype=float),
            feature_names=expected_names,
        )

        best_iteration = getattr(booster, "best_iteration", None)
        if best_iteration is None:
            probabilities = booster.predict(model_input)
        else:
            probabilities = booster.predict(
                model_input,
                iteration_range=(0, best_iteration + 1),
            )

        probability = float(probabilities[0])
        predicted_failure = int(probability >= 0.50)

        return {
            "equipment_id": equipment_ids[0],
            "failure_probability": probability,
            "predicted_failure": predicted_failure,
            "prediction_label": "Failure" if predicted_failure else "No Failure",
        }

    return predict_reading