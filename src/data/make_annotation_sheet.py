"""Scraped articles -> one labelling sheet per annotator in data/annotations/ (first N_OVERLAP shared by everyone)."""
import pandas as pd
from config import *

def main(seed=42):
    s = pd.read_json(SCRAPED_JSONL, lines=True)
    s = s[s.ok & s.chars.between(300, 6000)]
    meta = pd.read_csv(INTERIM/"gdelt_urls_sample.csv")[["article_id","date","state"]]
    s = s.merge(meta, on="article_id").sample(frac=1, random_state=seed).head(N_REAL_TEST).reset_index(drop=True)
    s["text_excerpt"] = s.text.str.slice(0, 1500)
    s["overlap"] = (s.index < N_OVERLAP).astype(int)
    blank = ["is_unrest_event","event_type","event_status","state_in_text","cause","notes"]
    for name in ANNOTATORS:
        mine = s[(s.overlap == 1) | (s.index % len(ANNOTATORS) == ANNOTATORS.index(name))].copy()
        for c in blank: mine[c] = ""
        mine = mine.rename(columns={"state": "gdelt_state"})
        mine[["article_id","overlap","url","date","gdelt_state","text_excerpt"] + blank] \
            .to_csv(ANN/f"to_label_{name}.csv", index=False)
        print(name, len(mine), "rows")

if __name__ == "__main__": main()
