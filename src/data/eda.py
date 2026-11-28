"""Save EDA plots. Run from repo root after preprocess.py:
    python src/data/eda.py --data data/processed --out reports/figures
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/processed")
ap.add_argument("--out", default="reports/figures")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

df = pd.concat([pd.read_csv(Path(a.data) / f"{s}.csv") for s in ["train", "val", "test"]])
ev = df[df.label_event_mention == "event_report"]

def bar(series, title, name, rot=0):
    ax = series.plot(kind="bar", figsize=(7, 4), title=title, rot=rot)
    ax.set_ylabel("count")
    plt.tight_layout(); plt.savefig(out / name, dpi=150); plt.close()

bar(df.label_event_mention.value_counts(), "Event vs not-event", "mention.png")
bar(ev.event_type.value_counts(), "Event type", "event_type.png")
bar(ev.event_status.value_counts(), "Event status", "event_status.png")
bar(df.groupby(["split", "event_type"]).size().unstack(fill_value=0), "Event type per split", "split_type.png")
bar(df.country.value_counts(), "Records per country", "country.png", rot=45)
bar(ev.groupby("year").size(), "Events per year", "year.png")
df.n_tokens.plot(kind="hist", bins=25, figsize=(7, 4), title="Text length (tokens)")
plt.tight_layout(); plt.savefig(out / "length.png", dpi=150); plt.close()
print("Saved plots to", out)
