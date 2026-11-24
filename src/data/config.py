"""Shared paths and settings for the data pipeline. Run all scripts from the repo root."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW, INTERIM, PROCESSED = ROOT/"data/raw", ROOT/"data/interim", ROOT/"data/processed"
ANN, REPORTS = ROOT/"data/annotations", ROOT/"reports"
for p in (RAW, INTERIM, PROCESSED, ANN, REPORTS): p.mkdir(parents=True, exist_ok=True)

ACLED_XLSX = RAW/"acled_asia_pacific_weekly.xlsx"
GDELT_CSV  = RAW/"gdelt_india_protest_2018_2023.csv"
SYNTH_CSV  = RAW/"synthetic_civil_unrest_news.csv"
REGION_WEEKLY = PROCESSED/"region_weekly.csv"
SCRAPED_JSONL = INTERIM/"articles_scraped.jsonl"

# GDELT uses old state names; map them to ACLED's ADMIN1 names
STATE_ALIASES = {"Uttaranchal":"Uttarakhand","Orissa":"Odisha","Pondicherry":"Puducherry",
                 "Dadra and Nagar Haveli":"Dadra and Nagar Haveli and Daman and Diu"}
JUNK_DOMAINS = {"prokerala.com"}          # add more after looking at reports/gdelt_domains.csv
ANNOTATORS = ["socion", "teammate2", "teammate3", "teammate4"]   # <- put real names
N_REAL_TEST, N_OVERLAP = 220, 20          # articles to label; first N_OVERLAP are labelled by everyone

# Real escalation label rule (agree on these with the team, then freeze them)
HIGH_MIN_EVENTS, HIGH_RATIO, MED_MIN_EVENTS, MED_RATIO = 5, 1.5, 2, 1.1

def to_acled_week(dates: pd.Series) -> pd.Series:
    """ACLED weeks start on Saturday: return the Saturday on or before each date."""
    d = pd.to_datetime(dates)
    return (d - pd.to_timedelta((d.dt.weekday - 5) % 7, unit="D")).dt.normalize()
