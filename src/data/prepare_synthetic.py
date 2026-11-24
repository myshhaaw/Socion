"""Validate the synthetic CSV and copy it to data/processed/articles_synthetic.csv."""
import pandas as pd
from config import *

REQUIRED = ["article_id","date","text","text_type","country","state","is_unrest_event","event_type","event_status",
            "trigger_phrase","cause","escalation_risk","state_protest_riot_prev4w","state_trend_ratio","split"]
ALLOWED = {"event_type": {"none","protest","riot","strike","march","blockade"},
           "event_status": {"none","past","ongoing","planned"},
           "escalation_risk": {"none","low","medium","high"}, "split": {"train","val","test"}}

def main():
    d = pd.read_csv(SYNTH_CSV)
    assert not (set(REQUIRED) - set(d.columns)), f"missing columns: {set(REQUIRED)-set(d.columns)}"
    assert d.article_id.is_unique and not d.text.duplicated().any(), "duplicate ids or texts"
    for col, ok in ALLOWED.items(): assert set(d[col].unique()) <= ok, f"bad values in {col}"
    pos = d[d.is_unrest_event == 1]
    assert pos.apply(lambda r: r.trigger_phrase in r.text, axis=1).all(), "trigger not found in text"
    assert (d[d.is_unrest_event == 0].event_type == "none").all()
    d.to_csv(PROCESSED/"articles_synthetic.csv", index=False)
    print("OK", d.shape, d.is_unrest_event.value_counts().to_dict(), d.split.value_counts().to_dict())

if __name__ == "__main__": main()
