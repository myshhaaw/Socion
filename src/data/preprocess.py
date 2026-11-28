"""
Socion - preprocessing for the synthetic civil unrest dataset.

Place at:  src/data/preprocess.py
Run from repo root:
    python src/data/preprocess.py --raw data/raw/synthetic_civil_unrest_review.csv --out data/processed

Produces (in --out):
    train.csv / val.csv / test.csv   cleaned, model-ready tables (provided split kept)
    train.jsonl / val.jsonl / test.jsonl   word-level BIO trigger tags for token classification
    country_month_counts.csv         event counts per country x month (input for forecasting)
    label_maps.json                  label <-> id maps
    preprocess_report.json           sanity stats to quote in the report
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

EXPECTED_COLS = [
    "record_id", "text", "source_type", "text_type", "date", "country", "region",
    "label_event_mention", "event_type", "event_status", "evidence_span",
    "negative_reason", "active_unrest_now", "review_status", "review_flag",
    "split", "provenance",
]
# Columns that describe the dataset's review process, not the text. Never feed to a model.
DROP_FOR_MODEL = ["review_status", "review_flag", "provenance"]

TRAILING_TAGS = re.compile(r"((?:\s+#\w+)+)\s*$")
WORD = re.compile(r"\w+|[^\w\s]")

SCALE_RE = re.compile(
    r"\b(?:(?:more than|nearly|over|around|about|at least|some)\s+)?"
    r"(?:\d[\d,]*|thousands|hundreds|dozens|scores)(?:\s+of)?\b",
    re.I,
)
CAUSE_RE = re.compile(r"\b(?:amid anger over|over|against|demanding)\s+(.+?)(?:[.!,]|$)", re.I)
TIME_RE = re.compile(
    r"\b(?:on (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"later this week|this (?:week|weekend)|tomorrow|today|on \w+day)\b", re.I
)


# ---------------------------------------------------------------- 1. load + validate
def load_and_validate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLS) - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    assert df.record_id.is_unique, "Duplicate record_id"
    assert df.text.notna().all(), "Null text"
    assert set(df.split.unique()) <= {"train", "val", "test"}
    # logical consistency between label columns
    neg = df.label_event_mention == "not_event_report"
    assert (df.loc[neg, "event_type"] == "none").all()
    assert (df.loc[~neg, "event_type"] != "none").all()
    assert (df.loc[~neg, "evidence_span"].notna()).all(), "event_report without evidence_span"
    return df


# ---------------------------------------------------------------- 2. dates
def add_dates(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="raise")  # source is DD-MM-YYYY
    df["date_iso"] = dt.dt.strftime("%Y-%m-%d")
    df["year"] = dt.dt.year
    df["year_month"] = dt.dt.to_period("M").astype(str)
    df["year_week"] = dt.dt.to_period("W").astype(str)
    return df


# ---------------------------------------------------------------- 3. text cleaning
def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    def split_tags(t: str):
        t = re.sub(r"\s+", " ", t).strip()
        m = TRAILING_TAGS.search(t)
        if not m:
            return t, []
        return t[: m.start()].strip(), re.findall(r"#(\w+)", m.group(1))

    out = df["text"].map(split_tags)
    df["text_clean"] = out.map(lambda x: x[0])      # text with trailing hashtag block removed
    df["hashtags"] = out.map(lambda x: x[1])        # kept separately as a feature/ablation
    df["n_hashtags"] = df["hashtags"].map(len)
    df["n_tokens"] = df["text_clean"].map(lambda s: len(WORD.findall(s)))
    return df


# ---------------------------------------------------------------- 4. trigger offsets + BIO
def locate_trigger(row):
    span = row["evidence_span"]
    if pd.isna(span):
        return pd.Series({"trig_start": -1, "trig_end": -1})
    i = row["text_clean"].lower().find(span.lower())
    assert i >= 0, f"Span not found in text_clean for {row['record_id']}"
    return pd.Series({"trig_start": i, "trig_end": i + len(span)})


def bio_tags(text: str, start: int, end: int):
    tokens, tags, inside = [], [], False
    for m in WORD.finditer(text):
        tokens.append(m.group())
        if start >= 0 and m.start() >= start and m.end() <= end:
            tags.append("I-TRIG" if inside else "B-TRIG")
            inside = True
        else:
            tags.append("O")
    return tokens, tags


# ---------------------------------------------------------------- 5. silver argument labels
def silver_arguments(row):
    """Rule-based silver labels for scale / actor / cause / time. NOT gold - verify a sample by hand."""
    if row["label_event_mention"] != "event_report":
        return pd.Series({"scale_text": None, "actor_text": None, "cause_text": None, "time_text": None})
    t = row["text_clean"]
    s = row["trig_start"]
    before = t[:s]
    before = re.sub(r"^(?:[^:!]{1,30}[:!]\s*)+", "", before).strip()   # drop 'BREAKING:' / 'Region:' prefixes
    sc = SCALE_RE.search(before)
    scale = sc.group().strip() if sc else None
    actor = SCALE_RE.sub("", before).strip(" ,") or None
    cause = CAUSE_RE.search(t[s:])
    tm = TIME_RE.search(t)
    return pd.Series({
        "scale_text": scale,
        "actor_text": actor,
        "cause_text": cause.group(1).strip() if cause else None,
        "time_text": tm.group().strip() if tm else None,
    })


# ---------------------------------------------------------------- 6. labels
def build_label_maps(df):
    maps = {
        "event_mention": {"not_event_report": 0, "event_report": 1},
        "event_type": {k: i for i, k in enumerate(sorted(df.event_type.unique()))},
        "event_status": {k: i for i, k in enumerate(sorted(df.event_status.unique()))},
        "bio": {"O": 0, "B-TRIG": 1, "I-TRIG": 2},
    }
    df["y_mention"] = df.label_event_mention.map(maps["event_mention"])
    df["y_type"] = df.event_type.map(maps["event_type"])
    df["y_status"] = df.event_status.map(maps["event_status"])
    # Punjab exists in both India and Pakistan -> always key a region by (country, region)
    df["region_key"] = df["country"] + "|" + df["region"]
    return df, maps


# ---------------------------------------------------------------- 7. forecasting table
def country_month_table(df):
    ev = df[df.label_event_mention == "event_report"]
    t = ev.pivot_table(index=["country", "year_month"], columns="event_type",
                       values="record_id", aggfunc="count", fill_value=0)
    t["total_events"] = t.sum(axis=1)
    t["n_ongoing"] = ev[ev.event_status == "ongoing"].groupby(["country", "year_month"]).size()
    t["n_planned"] = ev[ev.event_status == "planned"].groupby(["country", "year_month"]).size()
    t = t.fillna(0).astype(int).reset_index()
    # make the grid complete (months with zero events must appear as zeros)
    months = pd.period_range(df.year_month.min(), df.year_month.max(), freq="M").astype(str)
    grid = pd.MultiIndex.from_product([sorted(df.country.unique()), months], names=["country", "year_month"])
    t = t.set_index(["country", "year_month"]).reindex(grid, fill_value=0).reset_index()
    return t


# ---------------------------------------------------------------- main
def main(raw, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_and_validate(raw)
    df = add_dates(df)
    df = clean_text(df)
    df = pd.concat([df, df.apply(locate_trigger, axis=1)], axis=1)
    df = pd.concat([df, df.apply(silver_arguments, axis=1)], axis=1)
    df, maps = build_label_maps(df)

    # BIO tags
    bt = df.apply(lambda r: bio_tags(r["text_clean"], r["trig_start"], r["trig_end"]), axis=1)
    df["tokens"] = bt.map(lambda x: x[0])
    df["bio_tags"] = bt.map(lambda x: x[1])
    assert (df.loc[df.label_event_mention == "event_report", "bio_tags"]
            .map(lambda t: "B-TRIG" in t)).all(), "Some event has no trigger tag"

    model_cols = [c for c in df.columns if c not in DROP_FOR_MODEL]
    for name, part in df.groupby("split"):
        part[model_cols].to_csv(out / f"{name}.csv", index=False)
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part.itertuples():
                f.write(json.dumps({
                    "id": r.record_id, "tokens": r.tokens, "bio_tags": r.bio_tags,
                    "event_type": r.event_type, "event_status": r.event_status,
                    "country": r.country, "region": r.region, "date": r.date_iso,
                }) + "\n")

    country_month_table(df).to_csv(out / "country_month_counts.csv", index=False)
    json.dump(maps, open(out / "label_maps.json", "w"), indent=2)

    ev = df[df.label_event_mention == "event_report"]
    report = {
        "n_records": len(df),
        "split_sizes": df.split.value_counts().to_dict(),
        "label_event_mention": df.label_event_mention.value_counts().to_dict(),
        "event_type": df.event_type.value_counts().to_dict(),
        "event_status": df.event_status.value_counts().to_dict(),
        "date_range": [df.date_iso.min(), df.date_iso.max()],
        "mean_tokens": round(df.n_tokens.mean(), 1),
        "max_tokens": int(df.n_tokens.max()),
        "silver_scale_coverage": round(ev.scale_text.notna().mean(), 3),
        "silver_actor_coverage": round(ev.actor_text.notna().mean(), 3),
        "silver_cause_coverage": round(ev.cause_text.notna().mean(), 3),
        "silver_time_coverage": round(ev.time_text.notna().mean(), 3),
        "mean_events_per_country_month": round(len(ev) / (df.country.nunique() * df.year_month.nunique()), 2),
    }
    json.dump(report, open(out / "preprocess_report.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/synthetic_civil_unrest_review.csv")
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    main(a.raw, a.out)
