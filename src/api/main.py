
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "artifacts" / "forecasting" / "logistic_regression.joblib"

app = FastAPI(
    title="SOCION API",
    description="API for state-level civil unrest event forecasting",
    version="1.0.0"
)


model = None


class PredictionRequest(BaseModel):
    state: str = Field(min_length=1)

    acled_1w: float
    acled_4w: float
    acled_12w: float
    acled_trend: float
    acled_active_weeks_12w: float
    fatalities_4w: float
    gdelt_1w: float
    gdelt_4w: float
    gdelt_mentions_4w: float
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

        return {
            "status": "healthy",
            "model": "logistic_regression"
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
