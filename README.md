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
- Processed dataset: `data/processed/forecasting_dataset.csv`

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

Interactive API documentation is available at:

`http://127.0.0.1:8000/docs`

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Check whether the API is running |
| GET | `/health` | Check model and dataset availability and latest feature date |
| POST | `/predict` | Generate a seven-day event probability estimate using supplied features |
| GET | `/predict/latest?state=delhi` | Generate an estimate using the latest available historical features for a state |
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

### Example Prediction Response

```json
{
  "state": "delhi",
  "forecast_horizon_days": 7,
  "event_probability": 0.9507,
  "interpretation": "Estimated probability of a qualifying event in the next 7 days"
}
```

The example input and output are for demonstration purposes only.

### Latest Historical Prediction

The `/predict/latest` endpoint automatically retrieves the latest available feature record for the requested state from:

`data/processed/forecasting_dataset.csv`

Example request:

```http
GET /predict/latest?state=delhi
```

Example response:

```json
{
  "state": "delhi",
  "feature_date": "2023-12-23",
  "forecast_horizon_days": 7,
  "event_probability": 0.9438,
  "data_status": "historical",
  "warning": "The latest available dataset ends on 2023-12-23. This is not a live forecast."
}
```

The current processed dataset ends on 2023-12-23. Therefore, this endpoint provides a historical-data-based estimate, not a live forecast.

### Health Check

The `/health` endpoint checks whether the forecasting model and processed dataset are available.

Example response:

```json
{
  "status": "healthy",
  "model": "logistic_regression",
  "model_available": true,
  "dataset_available": true,
  "latest_feature_date": "2023-12-23"
}
```

### Run Tests

Run the API tests using:

```bash
python -m pytest test_api.py -v
```

## Important Limitations

- The `/predict` endpoint accepts precomputed aggregate features.
- The `/predict/latest` endpoint retrieves the latest available historical features from the processed forecasting dataset.
- The API does not fetch live data or automatically update the forecasting dataset.
- The current processed forecasting dataset ends on 2023-12-23; predictions must not be interpreted as current real-time conditions.
- Predictions are model-generated estimates, not confirmation that an event will occur.
- Model performance on sample data does not establish real-world forecasting accuracy.
- Results should be interpreted at an aggregate level and not used to identify or profile individuals.

## Ethical Considerations

SOCION aims to preserve privacy and responsible use of public data.

- Avoid identifying individual protesters or organizers.
- Do not create individual-level political or behavioural profiles.
- Preserve anonymization and respect applicable data licensing and access restrictions.
- Treat model predictions as estimates rather than verified events.
