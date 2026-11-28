
from pathlib import Path
import json

import joblib
import numpy as np
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


def evaluate_sequence_models():

    data = prepare_data()

    X_test = data["X_test"].copy()
    y_test = data["y_test"].copy()

    all_metrics = {}

    for model_type in ["GRU", "LSTM"]:

        print(f"\nEvaluating {model_type}...")

        model_path = MODEL_DIR / f"{model_type.lower()}_model.pth"
        scaler_path = MODEL_DIR / f"{model_type.lower()}_scaler.joblib"

        checkpoint = torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True,
        )

        scaler = joblib.load(scaler_path)

        X_scaled = X_test.copy()

        X_scaled[FEATURES] = scaler.transform(
            X_scaled[FEATURES]
        )

        X_sequences, y_sequences = create_sequences(
            X_scaled,
            y_test,
            SEQUENCE_LENGTH,
        )

        if len(X_sequences) == 0:
            raise ValueError("No test sequences generated.")

        if checkpoint["features"] != FEATURES:
            raise ValueError("Saved model features do not match.")

        if checkpoint["sequence_length"] != SEQUENCE_LENGTH:
            raise ValueError("Sequence length mismatch.")

        if checkpoint["hidden_size"] != HIDDEN_SIZE:
            raise ValueError("Hidden size mismatch.")

        model = SequenceModel(model_type).to(DEVICE)

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.eval()

        with torch.no_grad():

            logits = model(
                torch.tensor(X_sequences, dtype=torch.float32).to(DEVICE)
            )

            probabilities = torch.sigmoid(
                logits
            ).cpu().numpy()

        predictions = (probabilities >= 0.5).astype(int)

        metrics = {
            "model": model_type,
            "test_pr_auc": float(
                average_precision_score(y_sequences, probabilities)
            ),
            "test_precision": float(
                precision_score(y_sequences, predictions, zero_division=0)
            ),
            "test_recall": float(
                recall_score(y_sequences, predictions, zero_division=0)
            ),
            "test_f1": float(
                f1_score(y_sequences, predictions, zero_division=0)
            ),
            "test_brier_score": float(
                brier_score_loss(y_sequences, probabilities)
            ),
            "test_samples": int(len(y_sequences)),
            "threshold": 0.5,
        }

        tn, fp, fn, tp = confusion_matrix(
            y_sequences,
            predictions,
            labels=[0, 1],
        ).ravel()

        metrics["confusion_matrix"] = {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }

        all_metrics[model_type] = metrics

        for key, value in metrics.items():
            print(f"{key}: {value}")

    output_path = MODEL_DIR / "sequence_models_test_metrics.json"

    with open(output_path, "w") as file:
        json.dump(all_metrics, file, indent=4)

    print("\nTest comparison saved:", output_path)


if __name__ == "__main__":
    evaluate_sequence_models()
