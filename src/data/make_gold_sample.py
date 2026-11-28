import pandas as pd

df = pd.read_csv("data/processed/train.csv")
ev = df[df.label_event_mention == "event_report"]
cols = ["record_id", "text_clean", "evidence_span",
        "actor_text", "scale_text", "cause_text", "time_text"]
s = ev.sample(100, random_state=42)[cols].copy()
for c in ["actor_ok", "scale_ok", "cause_ok", "time_ok"]:
    s[c] = ""
s.to_csv("data/processed/gold_sample_to_check.csv", index=False)
print("saved", len(s), "rows")