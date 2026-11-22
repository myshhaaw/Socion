import pandas as pd
from src.config import RAW
from src.data.preprocess import save_interim

def _read(name):
    df = pd.read_csv(RAW / "CUT" / name, dtype={"tweet_id": str})
    df["tweet_id"] = df["tweet_id"].str.strip().str.strip('"')
    return df.drop_duplicates("tweet_id")

def load_cut():
    m = _read("majority_annotation_results.csv")
    m = m.drop(columns=["Unnamed: 0"], errors="ignore").rename(
        columns={"particiation": "participation", "country_code": "country"})
    m["year"] = m["year"].astype("Int64")

    gen = _read("labelled_tweets_is_general_unrest.csv").rename(
        columns={"label": "label_general_unrest"})
    prot = _read("labelled_tweets_is_protest_event.csv").rename(
        columns={"label": "label_protest_event"})

    df = m.merge(gen, on="tweet_id", how="left").merge(prot, on="tweet_id", how="left")
    df["post_id"] = df["tweet_id"]
    df["source"] = "CUT"
    df["text"] = pd.NA            # text is not distributed with CUT
    df["has_text"] = False
    df["in_india"] = df["country"].eq("IN")   # will be False for every row
    return df

if __name__ == "__main__":
    df = load_cut()
    print(f"rows: {len(df):,} | unique ids: {df.tweet_id.nunique():,}")
    print("has_text:", df.has_text.sum(), "| India rows:", df.in_india.sum())
    print(df["is_event"].value_counts(dropna=False))
    print(df["country"].value_counts().head(10))
    save_interim(df, "cut_cleaned", "cut_clean.csv")