
from pathlib import Path
import json

import joblib
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
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


def train_model():

    # Load prepared train, validation and test data
    data = prepare_data()

    X_train = data["X_train"]
    y_train = data["y_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]
    # Preprocess numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


    # Logistic Regression pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    # Train model
    print("\nTraining Logistic Regression...")

    model.fit(X_train, y_train)

    # Validation predictions
    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    # Evaluation metrics
    metrics = {
        "model": "Logistic Regression",
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

    # Save trained model
    model_path = MODEL_DIR / "logistic_regression.joblib"
    joblib.dump(model, model_path)

    # Save evaluation metrics
    metrics_path = MODEL_DIR / "logistic_regression_metrics.json"

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    # Display results
    print("\n========== LOGISTIC REGRESSION RESULTS ==========")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nModel saved:", model_path)
    print("Metrics saved:", metrics_path)

    print("\nLogistic Regression training completed!")


if __name__ == "__main__":
    train_model()
