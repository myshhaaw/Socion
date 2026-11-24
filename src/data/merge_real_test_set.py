"""Labelled sheets (data/annotations/labeled_<name>.csv) -> data/processed/real_test_set.csv + agreement report."""
import glob, pandas as pd
from sklearn.metrics import cohen_kappa_score
from config import *

TYPES = {"protest","riot","strike","march","blockade","none"}; STATUS = {"past","ongoing","planned","none"}

def main():
    parts = []
    for p in glob.glob(str(ANN/"labeled_*.csv")):
        d = pd.read_csv(p, dtype=str).fillna(""); d["annotator"] = Path(p).stem.replace("labeled_", ""); parts.append(d)
    d = pd.concat(parts)
    d = d[d.is_unrest_event.isin(["0", "1"])]
    bad = d[(d.is_unrest_event == "1") & (~d.event_type.isin(TYPES - {"none"}) | ~d.event_status.isin(STATUS - {"none"}))]
    assert bad.empty, f"invalid labels:\n{bad[['article_id','annotator','event_type','event_status']]}"
    # agreement on the shared items
    ov = d[d.overlap == "1"]; rep = ["# Annotation report", f"labelled rows: {len(d)}, articles: {d.article_id.nunique()}"]
    names = sorted(ov.annotator.unique())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            m = ov[ov.annotator == names[i]].merge(ov[ov.annotator == names[j]], on="article_id", suffixes=("_a", "_b"))
            if len(m) > 3:
                rep.append(f"{names[i]} vs {names[j]} (n={len(m)}): kappa is_unrest = "
                           f"{cohen_kappa_score(m.is_unrest_event_a, m.is_unrest_event_b):.2f}, "
                           f"kappa event_type = {cohen_kappa_score(m.event_type_a, m.event_type_b):.2f}")
    # one row per article: majority label for shared items
    mode = lambda s: s.mode().iloc[0]
    g = d.groupby("article_id").agg(is_unrest_event=("is_unrest_event", mode), event_type=("event_type", mode),
        event_status=("event_status", mode), cause=("cause", lambda s: next((x for x in s if x), "")),
        state_in_text=("state_in_text", lambda s: next((x for x in s if x), "")),
        n_annotators=("annotator", "nunique")).reset_index()
    g.loc[g.is_unrest_event == "0", ["event_type", "event_status"]] = "none"
    meta = pd.read_csv(INTERIM/"gdelt_urls_sample.csv")[["article_id","url","date","country","state","avg_tone"]]
    g = g.merge(meta, on="article_id")
    g["week"] = to_acled_week(g.date)
    rw = pd.read_csv(REGION_WEEKLY, parse_dates=["week"])[["country","admin1","week","prev4w","trend_ratio","real_escalation_label"]]
    g = g.merge(rw, left_on=["country","state","week"], right_on=["country","admin1","week"], how="left").drop(columns="admin1")
    g["escalation_risk"] = g.real_escalation_label.where(g.is_unrest_event == "1", "none")
    texts = pd.read_json(SCRAPED_JSONL, lines=True)[["article_id", "text"]]
    g = g.merge(texts, on="article_id"); g["is_unrest_event"] = g.is_unrest_event.astype(int)
    g.drop(columns=["real_escalation_label"]).to_csv(PROCESSED/"real_test_set.csv", index=False)
    rep.append(f"real_test_set rows: {len(g)}; unrest events: {int(g.is_unrest_event.sum())}; "
               f"escalation label coverage: {g.escalation_risk.notna().mean():.0%}")
    (REPORTS/"annotation_report.md").write_text("\n\n".join(rep)); print("\n".join(rep))

if __name__ == "__main__": main()
