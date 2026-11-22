
from pathlib import Path
import json
import joblib

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
)

from src.forecasting.prepare_data import prepare_data

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATHS = {
    "Logistic Regression": ROOT / "artifacts" / "forecasting" / "logistic_regression.joblib",
    "Random Forest": ROOT / "artifacts" / "forecasting" / "random_forest.joblib",
}


def evaluate_model():

    data = prepare_data()

    X_test = data["X_test"]
    y_test = data["y_test"]

    all_metrics = {}

    for model_name, model_path in MODEL_PATHS.items():

        model = joblib.load(model_path)

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        metrics = {
            "test_pr_auc": float(
                average_precision_score(y_test, probabilities)
            ),
            "test_precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "test_recall": float(
                recall_score(y_test, predictions, zero_division=0)
            ),
            "test_f1": float(
                f1_score(y_test, predictions, zero_division=0)
            ),
            "test_brier_score": float(
                brier_score_loss(y_test, probabilities)
            ),
        }

        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        metrics["confusion_matrix"] = {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }

        all_metrics[model_name] = metrics

        print(f"\n========== {model_name.upper()} ==========")

        for key, value in metrics.items():
            print(f"{key}: {value}")

    metrics_path = ROOT / "artifacts" / "forecasting" / "model_comparison_test_metrics.json"

    with open(metrics_path, "w") as file:
        json.dump(all_metrics, file, indent=4)

    print("\nComparison metrics saved:", metrics_path)


if __name__ == "__main__":
    evaluate_model()
