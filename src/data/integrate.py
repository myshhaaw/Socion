
import numpy as np
import pandas as pd
from src.config import INTERIM, PROCESSED, WINDOW_DAYS, HORIZON_DAYS, TRAIN_END, VAL_END

WINDOW_WEEKS = max(WINDOW_DAYS // 7, 1)     # 4
HORIZON_WEEKS = max(HORIZON_DAYS // 7, 1)   # 1

def anchor_weekday(acled):
    """ACLED's week-start weekday; GDELT dates are bucketed onto the same weekly grid."""
    return int(acled["week"].dt.weekday.mode().iloc[0])

def to_week_start(dates, anchor):
    d = pd.to_datetime(dates).dt.normalize()
    return d - pd.to_timedelta((d.dt.weekday - anchor) % 7, unit="D")

def build_event_registry(acled, gdelt):
    anchor = anchor_weekday(acled)
    g = gdelt.copy()
    g["week"] = to_week_start(g["date"], anchor)

    q = acled[acled["qualifying"] & acled["state"].notna()]
    a_week = (q.groupby(["state", "week"])["events"].sum()
              .rename("acled_events_same_state_week").reset_index())

    ra = acled.rename(columns={"events": "n_events"}).copy()
    ra["record_level"] = "aggregate_weekly"
    ra["date"] = ra["week"]
    ra["evidence_url"] = None
    ra = ra.merge(a_week, on=["state", "week"], how="left")
    ra["link_status"] = "acled_record"

    rg = g.copy()
    rg["record_level"] = "event"
    rg["n_events"] = 1
    rg["fatalities"] = np.nan
    rg = rg.merge(a_week, on=["state", "week"], how="left")
    rg["acled_events_same_state_week"] = rg["acled_events_same_state_week"].fillna(0)
    rg["link_status"] = np.where(rg["state"].isna(), "no_state",
                         np.where(rg["acled_events_same_state_week"] > 0,
                                  "linked_state_week", "no_acled_match"))

    cols = ["source", "source_id", "record_level", "week", "date", "state", "event_category",
            "n_events", "fatalities", "acled_events_same_state_week", "link_status", "evidence_url"]
    reg = (pd.concat([ra.reindex(columns=cols), rg.reindex(columns=cols)], ignore_index=True)
           .sort_values(["week", "state", "source"]).reset_index(drop=True))
    reg.insert(0, "event_id", [f"SOC{i:07d}" for i in range(len(reg))])
    return reg

def _series(s, state, weeks):
    try:
        return s.loc[state].reindex(weeks)
    except KeyError:
        return pd.Series(np.nan, index=weeks)

def build_forecasting_dataset(acled, gdelt):
    """One row per (state, week). Features use ONLY weeks <= the row's week.
    Target = any qualifying ACLED event in the next HORIZON_WEEKS week(s)."""
    anchor = anchor_weekday(acled)
    q = acled[acled["qualifying"] & acled["state"].notna()]
    g = gdelt[gdelt["state"].notna()].copy()
    g["week"] = to_week_start(g["date"], anchor)
    weeks = pd.date_range(q["week"].min(), q["week"].max(), freq="7D")

    a_n = q.groupby(["state", "week"])["events"].sum()
    a_f = q.groupby(["state", "week"])["fatalities"].sum()
    g_n = g.groupby(["state", "week"]).size()
    g_m = g.groupby(["state", "week"])["num_mentions"].sum()
    g_t = g.groupby(["state", "week"])["avg_tone"].mean()

    W, H = WINDOW_WEEKS, HORIZON_WEEKS
    frames = []
    for state in sorted(q["state"].unique()):
        acled_1w = _series(a_n, state, weeks).fillna(0)       # absent row = zero events
        fatal_1w = _series(a_f, state, weeks).fillna(0)
        gd_1w = _series(g_n, state, weeks).fillna(0)
        gd_m = _series(g_m, state, weeks).fillna(0)
        tone = _series(g_t, state, weeks)

        f = pd.DataFrame(index=weeks)
        f["acled_1w"] = acled_1w
        f["acled_4w"] = acled_1w.rolling(W).sum()
        f["acled_12w"] = acled_1w.rolling(12).sum()
        f["acled_trend"] = f["acled_4w"] - f["acled_4w"].shift(W)
        f["acled_active_weeks_12w"] = (acled_1w > 0).rolling(12).sum()
        f["fatalities_4w"] = fatal_1w.rolling(W).sum()
        f["gdelt_1w"] = gd_1w
        f["gdelt_4w"] = gd_1w.rolling(W).sum()
        f["gdelt_mentions_4w"] = gd_m.rolling(W).sum()
        f["gdelt_tone_4w"] = tone.rolling(W, min_periods=1).mean().fillna(0)

        future = acled_1w.rolling(H).sum().shift(-H)           # weeks t+1 .. t+H
        f["target"] = (future > 0).where(future.notna())
        f.insert(0, "state", state)
        frames.append(f.rename_axis("week").reset_index())

    out = pd.concat(frames).dropna(subset=["acled_12w", "target"]).copy()
    out["target"] = out["target"].astype(int)

    # chronological split; rows whose target week crosses a split boundary are purged
    tr, va = pd.Timestamp(TRAIN_END), pd.Timestamp(VAL_END)
    target_end = out["week"] + pd.Timedelta(days=7 * H + 6)
    out["split"] = np.select(
        [target_end <= tr, (out["week"] > tr) & (target_end <= va), out["week"] > va],
        ["train", "val", "test"], default="purged")
    out = out[out["split"] != "purged"]
    return out.reset_index(drop=True)

if __name__ == "__main__":
    acled = pd.read_csv(INTERIM / "acled_cleaned/acled_clean.csv", parse_dates=["week"])
    gdelt = pd.read_csv(INTERIM / "gdelt_cleaned/gdelt_clean.csv", parse_dates=["date"])
    PROCESSED.mkdir(parents=True, exist_ok=True)

    reg = build_event_registry(acled, gdelt)
    reg.to_csv(PROCESSED / "event_registry.csv", index=False)
    reg.sample(min(500, len(reg)), random_state=0).to_csv(PROCESSED / "event_registry_sample.csv", index=False)

    fc = build_forecasting_dataset(acled, gdelt)
    fc.to_csv(PROCESSED / "forecasting_dataset.csv", index=False)
    fc.sample(min(500, len(fc)), random_state=0).to_csv(PROCESSED / "forecasting_dataset_sample.csv", index=False)
    print(f"registry: {len(reg):,} rows | forecasting: {len(fc):,} rows")
    print(fc.groupby("split")["target"].agg(["size", "mean"]))
