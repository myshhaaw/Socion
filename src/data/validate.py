
import pandas as pd
from src.config import PROCESSED, INTERIM, REPORTS, TRAIN_END, VAL_END
from src.data.integrate import WINDOW_WEEKS, HORIZON_WEEKS

def check_target_and_features(fc, acled, n=300, seed=0):
    """Recompute target and one feature independently from cleaned ACLED -> proves no leakage."""
    a = acled[acled["qualifying"] & acled["state"].notna()]
    for r in fc.sample(min(n, len(fc)), random_state=seed).itertuples():
        s = a[a["state"] == r.state]
        nxt = s[(s["week"] > r.week) & (s["week"] <= r.week + pd.Timedelta(days=7 * HORIZON_WEEKS))]
        assert int(nxt["events"].sum() > 0) == int(r.target), f"target mismatch {r.state} {r.week}"
        past = s[(s["week"] > r.week - pd.Timedelta(days=7 * WINDOW_WEEKS)) & (s["week"] <= r.week)]
        assert past["events"].sum() == r.acled_4w, f"feature mismatch {r.state} {r.week}"

def main():
    acled = pd.read_csv(INTERIM / "acled_cleaned/acled_clean.csv", parse_dates=["week"])
    gdelt = pd.read_csv(INTERIM / "gdelt_cleaned/gdelt_clean.csv", parse_dates=["date"])
    reg = pd.read_csv(PROCESSED / "event_registry.csv", parse_dates=["week", "date"])
    fc = pd.read_csv(PROCESSED / "forecasting_dataset.csv", parse_dates=["week"])

    assert reg["event_id"].is_unique
    assert not fc.duplicated(["state", "week"]).any()
    check_target_and_features(fc, acled)
    s = fc.groupby("split")["week"].agg(["min", "max"])
    assert s.loc["train", "max"] < s.loc["val", "min"] < s.loc["val", "max"] < s.loc["test", "min"]
    gap = pd.Timedelta(days=7 * HORIZON_WEEKS + 6)
    assert s.loc["train", "max"] + gap <= pd.Timestamp(TRAIN_END), "train target crosses boundary"
    assert s.loc["val", "max"] + gap <= pd.Timestamp(VAL_END), "val target crosses boundary"

    g_reg = reg[reg["record_level"] == "event"]
    lines = [
        "# Data Quality Report", "",
        f"- ACLED weekly rows: {len(acled):,} | states: {acled.state.nunique()} | "
        f"weeks: {acled.week.nunique()} ({acled.week.min().date()} to {acled.week.max().date()})",
        f"- GDELT records: {len(gdelt):,} | missing state: {gdelt.state.isna().mean():.1%}",
        f"- Registry rows: {len(reg):,} ({len(g_reg):,} GDELT events)",
        "- GDELT link status (state-week level):",
        *[f"  - {k}: {v:.1%}" for k, v in g_reg["link_status"].value_counts(normalize=True).items()],
        f"- Forecasting rows: {len(fc):,} | positive rate: {fc.target.mean():.1%}",
        "", "## Rows and positive rate per split", "",
        fc.groupby("split")["target"].agg(["size", "mean"]).to_markdown(),
        "", "## Checks passed",
        "- unique event IDs; no duplicate (state, week) rows",
        "- target and 4-week feature recomputed independently on 300 random rows",
        "- train < val < test; no target window crosses a split boundary",
        "", "## Limitations",
        "- ACLED file is aggregated: linking is state-week level, not event-level.",
        "- GDELT reflects news coverage, not necessarily real events.",
    ]
    (REPORTS / "data_quality_report.md").write_text("\n".join(lines))
    print("\n".join(lines))

if __name__ == "__main__":
    main()
