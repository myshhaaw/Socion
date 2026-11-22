# SOCION

**An NLP-Driven Framework for Civil Unrest Event Extraction and Temporal Forecasting from Social Media Data**

## Project Overview

SOCION is an AI-driven framework designed to extract civil unrest-related events from publicly available data and support aggregate, state-level temporal forecasting.

The project combines Natural Language Processing (NLP), data processing, and machine learning techniques to study event patterns and estimate the likelihood of qualifying events over a future time window.

## Forecasting API

SOCION provides a REST API built using FastAPI. It uses a trained Logistic Regression model to estimate the probability of a qualifying event in the next seven days.

### Requirements

- Python 3.12+
- Dependencies listed in `requirements.txt`
- Trained model: `artifacts/forecasting/logistic_regression.joblib`

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/myshhaaw/Socion.git
cd Socion
python -m pip install -r requirements.txt
```

### Run the API

From the project root directory, execute:

```bash
python -m uvicorn src.api.main:app --reload
```

The API will run locally at:

`http://127.0.0.1:8000`

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Check whether the API is running |
| GET | `/health` | Check API and model availability |
| POST | `/predict` | Generate a seven-day event probability estimate |
| GET | `/docs` | Open interactive API documentation |

### Prediction Request

Send a POST request to `/predict` with the following JSON structure:

```json
{
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
```

### Example Response

```json
{
  "state": "delhi",
  "forecast_horizon_days": 7,
  "event_probability": 0.9507,
  "interpretation": "Estimated probability of a qualifying event in the next 7 days"
}
```

The example input and output are for demonstration purposes only.

### Run Tests

Run the API tests using:

```bash
python -m pytest test_api.py -v
```

## Important Limitations

- The API currently accepts precomputed aggregate features. It does not fetch live data or calculate these features automatically.
- Predictions are model-generated estimates, not confirmation that an event will occur.
- Model performance on sample data does not establish real-world forecasting accuracy.
- Results should be interpreted at an aggregate level and not used to identify or profile individuals.

## Ethical Considerations

SOCION aims to preserve privacy and responsible use of public data.

- Avoid identifying individual protesters or organizers.
- Do not create individual-level political or behavioural profiles.
- Preserve anonymization and respect applicable data licensing and access restrictions.
- Treat model predictions as estimates rather than verified events.
