
from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)

from torch.utils.data import DataLoader, TensorDataset

from src.forecasting.prepare_data import prepare_data


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "artifacts" / "forecasting"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 12
HIDDEN_SIZE = 32
EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 0.001

DEVICE = torch.device("cpu")

FEATURES = [
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


class SequenceModel(nn.Module):

    def __init__(self, model_type):
        super().__init__()

        if model_type == "GRU":
            self.rnn = nn.GRU(
                input_size=len(FEATURES),
                hidden_size=HIDDEN_SIZE,
                batch_first=True,
            )

        else:
            self.rnn = nn.LSTM(
                input_size=len(FEATURES),
                hidden_size=HIDDEN_SIZE,
                batch_first=True,
            )

        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):

        result = self.rnn(x)

        if isinstance(result, tuple):
            output, _ = result
        else:
            output = result[0]

        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)

        return self.output(last_output).squeeze(1)


def create_sequences(X, y, sequence_length):

    sequences = []
    labels = []

    for state in X["state"].unique():

        state_indices = np.where(
            X["state"].values == state
        )[0]

        state_X = X.iloc[state_indices][FEATURES].values
        state_y = y.iloc[state_indices].values

        for i in range(sequence_length - 1, len(state_X)):

            sequences.append(
                state_X[i - sequence_length + 1:i + 1]
            )

            labels.append(state_y[i])

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
    )


def train_sequence_model(model_type, data):

    X_train = data["X_train"].copy()
    y_train = data["y_train"].copy()

    X_val = data["X_val"].copy()
    y_val = data["y_val"].copy()

    # Fit scaler only on training data
    scaler = StandardScaler()

    X_train[FEATURES] = scaler.fit_transform(
        X_train[FEATURES]
    )

    X_val[FEATURES] = scaler.transform(
        X_val[FEATURES]
    )

    # Build sequences
    Xtr, ytr = create_sequences(
        X_train, y_train, SEQUENCE_LENGTH
    )

    Xv, yv = create_sequences(
        X_val, y_val, SEQUENCE_LENGTH
    )

    if len(Xtr) == 0 or len(Xv) == 0:
        raise ValueError(
            "Not enough data to create sequences."
        )

    print(
        f"\n{model_type} sequences: "
        f"Train={len(Xtr)}, Validation={len(Xv)}"
    )

    train_dataset = TensorDataset(
        torch.tensor(Xtr),
        torch.tensor(ytr),
    )

    loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = SequenceModel(model_type).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Training
    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for xb, yb in loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            logits = model(xb)

            loss = criterion(logits, yb)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(
                f"{model_type} Epoch "
                f"{epoch + 1}/{EPOCHS}, "
                f"Loss={total_loss / len(loader):.4f}"
            )

    # Validation
    model.eval()

    with torch.no_grad():

        logits = model(
            torch.tensor(Xv).to(DEVICE)
        )

        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy()

    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": model_type,
        "validation_pr_auc": float(
            average_precision_score(yv, probabilities)
        ),
        "validation_precision": float(
            precision_score(yv, predictions, zero_division=0)
        ),
        "validation_recall": float(
            recall_score(yv, predictions, zero_division=0)
        ),
        "validation_f1": float(
            f1_score(yv, predictions, zero_division=0)
        ),
        "validation_brier_score": float(
            brier_score_loss(yv, probabilities)
        ),
        "validation_samples": int(len(yv)),
        "threshold": 0.5,
    }

    # Save model, scaler and metrics
    model_path = MODEL_DIR / f"{model_type.lower()}_model.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": model_type,
            "features": FEATURES,
            "sequence_length": SEQUENCE_LENGTH,
            "hidden_size": HIDDEN_SIZE,
        },
        model_path,
    )

    joblib.dump(
        scaler,
        MODEL_DIR / f"{model_type.lower()}_scaler.joblib",
    )

    metrics_path = MODEL_DIR / f"{model_type.lower()}_metrics.json"

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"\n========== {model_type} RESULTS ==========")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(f"\nModel saved: {model_path}")

    return metrics


def main():

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    data = prepare_data()

    results = []

    for model_type in ["GRU", "LSTM"]:

        result = train_sequence_model(
            model_type,
            data,
        )

        results.append(result)

    print("\n========== SEQUENCE MODEL COMPARISON ==========")

    for result in results:

        print(
            f"{result['model']}: "
            f"PR-AUC={result['validation_pr_auc']:.4f}, "
            f"F1={result['validation_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
