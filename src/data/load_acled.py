
import pandas as pd
from src.config import RAW, START_DATE, END_DATE, ACLED_QUALIFYING, COUNTRY
from src.data.preprocess import norm_state, save_interim, OFFICIAL_STATES

def load_acled(path=None):
    path = path or sorted((RAW / "ACLED").glob("*.csv"))[0]
    df = pd.read_csv(path, low_memory=False)
    df = df[df["COUNTRY"].astype(str).str.strip().str.lower().eq(COUNTRY.lower())].copy()

    out = pd.DataFrame({
        "source": "ACLED",
        "source_id": df["ID"].astype(str),
        "week": pd.to_datetime(df["WEEK"], errors="coerce").dt.normalize(),
        "state": df["ADMIN1"].map(norm_state),
        "event_type_raw": df["EVENT_TYPE"],
        "sub_event_type_raw": df["SUB_EVENT_TYPE"],
        "disorder_type": df["DISORDER_TYPE"],
        "events": pd.to_numeric(df["EVENTS"], errors="coerce").fillna(0).astype(int),
        "fatalities": pd.to_numeric(df["FATALITIES"], errors="coerce").fillna(0),
        "population_exposure": pd.to_numeric(df["POPULATION_EXPOSURE"], errors="coerce"),
        "latitude": df["CENTROID_LATITUDE"],
        "longitude": df["CENTROID_LONGITUDE"],
    })
    out["event_category"] = (out["event_type_raw"].map({"Protests": "protest", "Riots": "riot"})
                             .fillna("other"))
    out["qualifying"] = out["event_type_raw"].isin(ACLED_QUALIFYING)

    n_bad = out["week"].isna().sum()
    if n_bad:
        print(f"WARNING: {n_bad} rows with unparseable WEEK dropped")
    out = out.dropna(subset=["week"])
    # drop only exact duplicates of the aggregation key (not by ID, which may not be unique)
    out = out.drop_duplicates(["week", "state", "event_type_raw", "sub_event_type_raw"])
    out = out[out["week"].between(START_DATE, END_DATE)]

    unknown = sorted(set(out["state"].dropna()) - OFFICIAL_STATES)
    if unknown:
        print("WARNING: state names not in official list -> add to STATE_ALIASES:", unknown)
    return out.reset_index(drop=True)

if __name__ == "__main__":
    d = load_acled()
    print(f"rows: {len(d):,} | weeks: {d.week.nunique()} | states: {d.state.nunique()}")
    print(f"range: {d.week.min().date()} -> {d.week.max().date()}")
    print("week start weekday:", d.week.dt.day_name().value_counts().to_dict())
    print(d.groupby("event_type_raw")["events"].sum())
    save_interim(d, "acled_cleaned", "acled_clean.csv")
