
from pathlib import Path
import json

import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
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

MODEL_DIR = ROOT / "artifacts" / "forecasting"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_FEATURES = [
    "acled_1w",
    "acled_4w",
    "acled_12w",
    "acled_trend",
    "acled_active_weeks_12w",
    "fatalities_4w",
    "gdelt_1w",
    "gdelt_4w",
    "gdelt_mentions_4w",
    "gdelt_tone_4w",
]

CATEGORICAL_FEATURES = ["state"]


def train_random_forest():

    data = prepare_data()

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_split=5,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("\nTraining Random Forest...")

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": "Random Forest",
        "validation_pr_auc": float(
            average_precision_score(y_val, probabilities)
        ),
        "validation_precision": float(
            precision_score(y_val, predictions, zero_division=0)
        ),
        "validation_recall": float(
            recall_score(y_val, predictions, zero_division=0)
        ),
        "validation_f1": float(
            f1_score(y_val, predictions, zero_division=0)
        ),
        "validation_brier_score": float(
            brier_score_loss(y_val, probabilities)
        ),
        "validation_positive_rate": float(y_val.mean()),
        "threshold": 0.5,
    }

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    metrics["confusion_matrix"] = {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    model_path = MODEL_DIR / "random_forest.joblib"
    metrics_path = MODEL_DIR / "random_forest_metrics.json"

    joblib.dump(model, model_path)

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    print("\n========== RANDOM FOREST RESULTS ==========")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nModel saved:", model_path)
    print("Metrics saved:", metrics_path)

    print("\nRandom Forest training completed!")


if __name__ == "__main__":
    train_random_forest()
