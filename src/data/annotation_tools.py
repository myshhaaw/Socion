import pandas as pd
from sklearn.metrics import cohen_kappa_score
from src.config import ANNOT, PROCESSED

def make_annotation_sample(df, n=800, overlap=0.15, seed=42,
                           annotators=("A", "B"), out=ANNOT / "to_annotate.csv"):
    """Sample posts; a share gets double-annotated for agreement measurement."""
    s = df.dropna(subset=["text"]).sample(n=min(n, len(df)), random_state=seed)
    s = s[["post_id", "text"]].copy()
    k = int(len(s) * overlap)
    shared, rest = s.iloc[:k], s.iloc[k:]
    half = len(rest) // 2
    parts = []
    for ann in annotators:                       # everyone labels the shared part
        parts.append(shared.assign(annotator=ann, double_annotated=True))
    parts.append(rest.iloc[:half].assign(annotator=annotators[0], double_annotated=False))
    parts.append(rest.iloc[half:].assign(annotator=annotators[1], double_annotated=False))
    res = pd.concat(parts)
    for c in ["event_type", "event_status", "trigger", "actor", "target",
              "location", "time_expr", "grievance"]:
        res[c] = ""
    res.to_csv(out, index=False)
    return res

def agreement(path=ANNOT / "annotated_posts.csv", col="event_type"):
    df = pd.read_csv(path)
    d = df[df["double_annotated"]]
    a = d[d.annotator == "A"].set_index("post_id")
    b = d[d.annotator == "B"].set_index("post_id")
    common = a.index.intersection(b.index)
    return cohen_kappa_score(a.loc[common, col], b.loc[common, col])