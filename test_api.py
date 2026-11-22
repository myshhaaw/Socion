
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["project"] == "SOCION"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "healthy"
    assert result["model"] == "logistic_regression"
    assert result["model_available"] is True
    assert result["dataset_available"] is True
    assert result["latest_feature_date"] == "2023-12-23"

def test_prediction():
    data = {
        "state": "delhi",
        "acled_1w": 5,
        "acled_4w": 11,
        "acled_12w": 64,
        "acled_trend": -12,
        "acled_active_weeks_12w": 12,
        "fatalities_4w": 0,
        "gdelt_1w": 2,
        "gdelt_4w": 10,
        "gdelt_mentions_4w": 50,
        "gdelt_tone_4w": -2.5
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200

    result = response.json()

    assert result["state"] == "delhi"
    assert result["forecast_horizon_days"] == 7
    assert 0 <= result["event_probability"] <= 1


def test_invalid_state():
    data = {
        "state": "",
        "acled_1w": 5,
        "acled_4w": 11,
        "acled_12w": 64,
        "acled_trend": -12,
        "acled_active_weeks_12w": 12,
        "fatalities_4w": 0,
        "gdelt_1w": 2,
        "gdelt_4w": 10,
        "gdelt_mentions_4w": 50,
        "gdelt_tone_4w": -2.5
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 422
def test_predict_latest():
    response = client.get("/predict/latest?state=delhi")

    assert response.status_code == 200

    result = response.json()

    assert result["state"] == "delhi"
    assert result["feature_date"] == "2023-12-23"
    assert result["forecast_horizon_days"] == 7
    assert 0 <= result["event_probability"] <= 1
    assert result["data_status"] == "historical"
    assert "not a live forecast" in result["warning"]


def test_predict_latest_invalid_state():
    response = client.get("/predict/latest?state=unknown_state")

    assert response.status_code == 404
def test_negative_event_count():
    data = {
        "state": "delhi",
        "acled_1w": -5,
        "acled_4w": 11,
        "acled_12w": 64,
        "acled_trend": -12,
        "acled_active_weeks_12w": 12,
        "fatalities_4w": 0,
        "gdelt_1w": 2,
        "gdelt_4w": 10,
        "gdelt_mentions_4w": 50,
        "gdelt_tone_4w": -2.5
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 422
