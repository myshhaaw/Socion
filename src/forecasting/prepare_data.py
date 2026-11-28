
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "forecasting_dataset.csv"

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
    "state",
]


def prepare_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["week"])

    required = FEATURES + ["target", "split"]
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.duplicated(["state", "week"]).any():
        raise ValueError("Duplicate state-week rows found")

    if df["target"].isna().any():
        raise ValueError("Missing target values found")

    X = df[FEATURES + ["week"]].copy()
    y = df["target"].astype(int)

    train = df["split"] == "train"
    val = df["split"] == "val"
    test = df["split"] == "test"

    return {
        "X_train": X[train].reset_index(drop=True),
        "y_train": y[train].reset_index(drop=True),
        "X_val": X[val].reset_index(drop=True),
        "y_val": y[val].reset_index(drop=True),
        "X_test": X[test].reset_index(drop=True),
        "y_test": y[test].reset_index(drop=True),
    }


if __name__ == "__main__":
    data = prepare_data()

    for split in ["train", "val", "test"]:
        print(
            f"{split.upper()}: "
            f"{len(data['X_' + split])} rows | "
            f"Positive rate: {data['y_' + split].mean():.4f}"
        )

    print("\nData preparation successful!")
