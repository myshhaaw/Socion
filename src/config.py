from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
ANNOT = ROOT / "data" / "annotations"
REPORTS = ROOT / "reports"

# ---- Scope (confirm with team, then document in data/README.md) ----
COUNTRY = "India"
START_DATE, END_DATE = "2018-01-01", "2023-12-31"
GEO_UNIT = "state"                      # GDELT/CUT can't reliably resolve districts
ACLED_QUALIFYING = ["Protests", "Riots"]

# ---- Forecasting design ----
WINDOW_DAYS = 28
HORIZON_DAYS = 7
MATCH_TOLERANCE_DAYS = 2                # event linking window

# ---- Chronological split ----
TRAIN_END = "2021-12-31"
VAL_END = "2022-12-31"                  # test = after this date