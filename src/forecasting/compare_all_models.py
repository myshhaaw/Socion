
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
)

from src.forecasting.prepare_data import prepare_data
from src.forecasting.sequence_models import (
    SequenceModel,
    create_sequences,
    FEATURES,
    SEQUENCE_LENGTH,
    HIDDEN_SIZE,
    DEVICE,
    MODEL_DIR,
)


def calculate_metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    return {
        "test_pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        "test_precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "test_recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "test_f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "test_brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "test_samples": int(len(y_true)),
        "threshold": 0.5,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def evaluate_all_models():

    data = prepare_data()

    X_test = data["X_test"].copy()
    y_test = data["y_test"].copy()

    # Generate sequence endpoints to identify the exact
    # 1,440 state-week rows used by GRU and LSTM.
    X_reference = X_test.copy()

    reference_sequences, reference_labels = create_sequences(
        X_reference,
        y_test,
        SEQUENCE_LENGTH,
    )

    if len(reference_sequences) == 0:
        raise ValueError("No reference sequences generated.")

    endpoint_indices = []

    for state in X_test["state"].unique():

        state_indices = np.where(
            X_test["state"].values == state
        )[0]

        state_weeks = pd.to_datetime(
            X_test.iloc[state_indices]["week"]
        ).reset_index(drop=True)

        for i in range(SEQUENCE_LENGTH - 1, len(state_indices)):

            window_weeks = state_weeks[
                i - SEQUENCE_LENGTH + 1:i + 1
            ]

            if (
                window_weeks.diff()
                .dropna()
                .dt.days.eq(7)
                .all()
            ):
                endpoint_indices.append(state_indices[i])

    if len(endpoint_indices) != len(reference_labels):
        raise ValueError("Endpoint alignment check failed.")

    # Keep only the exact weeks predicted by the sequence models.
    X_common = X_test.iloc[endpoint_indices].reset_index(drop=True)
    y_common = y_test.iloc[endpoint_indices].reset_index(drop=True)

    print("Common comparison samples:", len(y_common))
    print("Common comparison positive rate:", round(y_common.mean(), 4))

    all_metrics = {}

    # Logistic Regression and Random Forest
    traditional_models = {
        "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
        "Random Forest": MODEL_DIR / "random_forest.joblib",
    }

    for model_name, model_path in traditional_models.items():

        print(f"\nEvaluating {model_name}...")

        model = joblib.load(model_path)

        probabilities = model.predict_proba(X_common)[:, 1]

        metrics = calculate_metrics(y_common, probabilities)

        all_metrics[model_name] = metrics

        for key, value in metrics.items():
            print(f"{key}: {value}")

    # GRU and LSTM
    for model_type in ["GRU", "LSTM"]:

        print(f"\nEvaluating {model_type}...")

        checkpoint = torch.load(
            MODEL_DIR / f"{model_type.lower()}_model.pth",
            map_location=DEVICE,
            weights_only=True,
        )

        if checkpoint["features"] != FEATURES:
            raise ValueError("Saved model features do not match.")

        if checkpoint["sequence_length"] != SEQUENCE_LENGTH:
            raise ValueError("Sequence length mismatch.")

        if checkpoint["hidden_size"] != HIDDEN_SIZE:
            raise ValueError("Hidden size mismatch.")

        scaler = joblib.load(
            MODEL_DIR / f"{model_type.lower()}_scaler.joblib"
        )

        X_scaled = X_test.copy()

        X_scaled[FEATURES] = scaler.transform(
            X_scaled[FEATURES]
        )

        X_sequences, y_sequences = create_sequences(
            X_scaled,
            y_test,
            SEQUENCE_LENGTH,
        )

        if len(X_sequences) != len(y_common):
            raise ValueError("Sequence sample count mismatch.")

        if not np.array_equal(y_sequences, y_common.to_numpy()):
            raise ValueError("Sequence labels do not match common labels.")

        model = SequenceModel(model_type).to(DEVICE)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():

            logits = model(
                torch.tensor(
                    X_sequences,
                    dtype=torch.float32,
                ).to(DEVICE)
            )

            probabilities = torch.sigmoid(
                logits
            ).cpu().numpy()

        metrics = calculate_metrics(y_common, probabilities)

        all_metrics[model_type] = metrics

        for key, value in metrics.items():
            print(f"{key}: {value}")

    output_path = MODEL_DIR / "all_models_fair_comparison.json"

    with open(output_path, "w") as file:
        json.dump(all_metrics, file, indent=4)

    print("\nFair comparison saved:", output_path)


if __name__ == "__main__":
    evaluate_all_models()
