import pandas as pd
from src.config import RAW, START_DATE, END_DATE
from src.data.preprocess import norm_state, norm_place, save_interim

def parse_place(full):
    """'Jaipur, Rajasthan, India' -> (state, place)."""
    if pd.isna(full):
        return None, None
    parts = [p.strip() for p in str(full).split(",")]
    if len(parts) >= 3:
        return norm_state(parts[-2]), norm_place(parts[0])
    if len(parts) == 2:
        return norm_state(parts[0]), None
    return None, None

def categorize(code):
    code = str(code)
    if code.startswith("143"):
        return "strike"
    if code.startswith("145"):
        return "riot"
    return "protest"

def load_gdelt(path=None):
    path = path or sorted((RAW / "GDELT").glob("*.csv"))[0]
    df = pd.read_csv(path, dtype={"EventCode": str, "EventRootCode": str}, low_memory=False)
    df = df[(df["ActionGeo_CountryCode"] == "IN") & (df["EventRootCode"] == "14")].copy()
    geo = df["ActionGeo_FullName"].map(parse_place)
    out = pd.DataFrame({
        "source": "GDELT",
        "source_id": df["GLOBALEVENTID"].astype(str),
        "date": pd.to_datetime(df["SQLDATE"].astype(str), format="%Y%m%d", errors="coerce"),
        "state": geo.map(lambda x: x[0]),
        "place": geo.map(lambda x: x[1]),
        "latitude": df["ActionGeo_Lat"],
        "longitude": df["ActionGeo_Long"],
        "event_code": df["EventCode"],
        "event_category": df["EventCode"].map(categorize),
        "num_mentions": df["NumMentions"],
        "num_articles": df["NumArticles"],
        "avg_tone": df["AvgTone"],
        "goldstein": df["GoldsteinScale"],
        "evidence_url": df["SOURCEURL"],
    })
    out = out.dropna(subset=["source_id", "date"]).drop_duplicates("source_id")
    # the same article often produces repeated rows
    out = out.drop_duplicates(["evidence_url", "state", "date", "event_category"])
    return out[out["date"].between(START_DATE, END_DATE)]

if __name__ == "__main__":
    save_interim(load_gdelt(), "gdelt_cleaned", "gdelt_clean.csv")