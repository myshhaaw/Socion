"""
Baselines: keyword rules and TF-IDF + Logistic Regression.
Place at src/models/baselines.py. Run from repo root:
    python src/models/baselines.py
Writes reports/baseline_results.csv
"""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/processed")
ap.add_argument("--out", default="reports/baseline_results.csv")
a = ap.parse_args()

tr, va, te = [pd.read_csv(Path(a.data) / f"{s}.csv") for s in ["train", "val", "test"]]
for d in (tr, va, te):
    d["hashtags"] = d["hashtags"].fillna("[]")
    d["text_with_tags"] = d["text_clean"] + " " + d["hashtags"].str.replace(r"[\[\]',]", " ", regex=True)

# ---- keyword rule baseline -------------------------------------------------
TYPE_KW = {
    "protest": ["protest", "demonstration", "sit-in", "agitation", "rally"],
    "march": ["march", "rally", "procession"],
    "strike": ["strike", "walkout", "walk out", "shutdown"],
    "blockade": ["block", "road jam", "rail roko", "gherao"],
    "riot": ["riot", "clash", "violence", "arson", "set fire", "set vehicles on fire", "stone"],
}
ALL_KW = sorted({k for v in TYPE_KW.values() for k in v})

def rule_mention(text):
    t = text.lower()
    return "event_report" if any(k in t for k in ALL_KW) else "not_event_report"

def rule_type(text):
    t = text.lower()
    for typ in ["riot", "strike", "blockade", "march", "protest"]:   # most specific first
        if any(k in t for k in TYPE_KW[typ]):
            return typ
    return "none"

rows = []
def score(task, model, y_true, y_pred, split):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    rows.append(dict(task=task, model=model, split=split, accuracy=round(accuracy_score(y_true, y_pred), 3),
                     macro_precision=round(p, 3), macro_recall=round(r, 3), macro_f1=round(f, 3)))

for name, d in [("val", va), ("test", te)]:
    score("event_vs_not", "keyword_rules", d.label_event_mention, d.text_clean.map(rule_mention), name)
    score("event_type", "keyword_rules", d.event_type, d.text_clean.map(rule_type), name)

# ---- TF-IDF + logistic regression ------------------------------------------
tasks = {"event_vs_not": "label_event_mention", "event_type": "event_type", "event_status": "event_status"}
for task, ycol in tasks.items():
    for variant, col in [("tfidf_lr", "text_clean"), ("tfidf_lr_with_hashtags", "text_with_tags")]:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, lowercase=True)
        Xtr = vec.fit_transform(tr[col])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, tr[ycol])
        for name, d in [("val", va), ("test", te)]:
            score(task, variant, d[ycol], clf.predict(vec.transform(d[col])), name)

res = pd.DataFrame(rows)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
res.to_csv(a.out, index=False)
print(res[res.split == "test"].to_string(index=False))
print("\nSaved to", a.out)
