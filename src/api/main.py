
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "artifacts" / "forecasting" / "logistic_regression.joblib"
DATA_PATH = ROOT / "data" / "processed" / "forecasting_dataset.csv"

app = FastAPI(
    title="SOCION API",
    description="API for state-level civil unrest event forecasting",
    version="1.0.0"
)


model = None


class PredictionRequest(BaseModel):
    state: str = Field(min_length=1)

    acled_1w: float = Field(ge=0)
    acled_4w: float = Field(ge=0)
    acled_12w: float = Field(ge=0)
    acled_trend: float
    acled_active_weeks_12w: float = Field(ge=0, le=12)
    fatalities_4w: float = Field(ge=0)
    gdelt_1w: float = Field(ge=0)
    gdelt_4w: float = Field(ge=0)
    gdelt_mentions_4w: float = Field(ge=0)
    gdelt_tone_4w: float


def load_model():
    global model

    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Forecasting model not found")

        model = joblib.load(MODEL_PATH)

    return model


@app.get("/")
def home():
    return {
        "project": "SOCION",
        "message": "SOCION Forecasting API is running"
    }

@app.get("/health")
def health():
    try:
        load_model()

        if not DATA_PATH.exists():
            raise FileNotFoundError("Forecasting dataset not found")

        df = pd.read_csv(DATA_PATH, usecols=["week"])
        latest_date = pd.to_datetime(df["week"]).max()

        return {
            "status": "healthy",
            "model": "logistic_regression",
            "model_available": True,
            "dataset_available": True,
            "latest_feature_date": latest_date.strftime("%Y-%m-%d")
        }

    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        forecasting_model = load_model()

        input_data = pd.DataFrame([request.model_dump()])

        probability = forecasting_model.predict_proba(input_data)[0]

        positive_index = list(forecasting_model.classes_).index(1)

        risk_probability = float(probability[positive_index])

        return {
            "state": request.state,
            "forecast_horizon_days": 7,
            "event_probability": round(risk_probability, 4),
            "interpretation": (
                "Estimated probability of a qualifying event "
                "in the next 7 days"
            )
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/predict/latest")
def predict_latest(state: str):
    try:
        forecasting_model = load_model()

        if not DATA_PATH.exists():
            raise FileNotFoundError("Forecasting dataset not found")

        df = pd.read_csv(DATA_PATH, parse_dates=["week"])

        state = state.strip().lower()

        state_data = df[df["state"].str.lower() == state]

        if state_data.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No forecasting data found for state: {state}"
            )

        latest_row = state_data.sort_values("week").iloc[-1]

        feature_columns = [
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
            "state"
        ]

        input_data = pd.DataFrame([
            {column: latest_row[column] for column in feature_columns}
        ])

        probability = forecasting_model.predict_proba(input_data)[0]
        positive_index = list(forecasting_model.classes_).index(1)

        risk_probability = float(probability[positive_index])

        feature_date = latest_row["week"].strftime("%Y-%m-%d")

        return {
            "state": state,
            "feature_date": feature_date,
            "forecast_horizon_days": 7,
            "event_probability": round(risk_probability, 4),
            "data_status": "historical",
            "warning": (
                "The latest available dataset ends on "
                f"{feature_date}. This is not a live forecast."
            )
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
