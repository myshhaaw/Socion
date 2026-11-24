"""ACLED weekly file -> data/processed/region_weekly.csv  (one row per country/state/week, zeros filled)."""
import pandas as pd, numpy as np
from config import *

def main():
    a = pd.read_excel(ACLED_XLSX)
    a = a[a.EVENT_TYPE.isin(["Protests", "Riots"]) & (a.WEEK >= "2017-01-01")]
    key = ["COUNTRY", "ADMIN1", "WEEK"]
    w = (a.pivot_table(index=key, columns="EVENT_TYPE", values="EVENTS", aggfunc="sum", fill_value=0)
           .join(a.groupby(key).FATALITIES.sum()).reset_index())
    weeks = pd.date_range(w.WEEK.min(), w.WEEK.max(), freq="7D")
    out = []
    for (c, s), g in w.groupby(["COUNTRY", "ADMIN1"]):
        g = g.set_index("WEEK")[["Protests", "Riots", "FATALITIES"]].reindex(weeks, fill_value=0)
        tot = g.Protests + g.Riots
        prev4 = tot.shift(1).rolling(4, min_periods=1).sum()
        base = tot.shift(1).rolling(52, min_periods=8).mean()
        ratio = (prev4 / (base * 4)).where(base > 0, 1.0).clip(upper=5).fillna(1.0)
        next2 = tot.shift(-1) + tot.shift(-2)
        fat2 = g.FATALITIES.shift(-1) + g.FATALITIES.shift(-2)
        base2 = base * 2
        high = ((next2 >= np.maximum(HIGH_MIN_EVENTS, HIGH_RATIO * base2)) | (fat2 >= 1))
        med = next2 >= np.maximum(MED_MIN_EVENTS, MED_RATIO * base2)
        lab = pd.Series(np.where(high, "high", np.where(med, "medium", "low")), index=g.index)
        lab[next2.isna() | base.isna()] = np.nan      # unknown at the edges
        out.append(pd.DataFrame({"country": c, "admin1": s, "week": weeks, "protests": g.Protests.values,
            "riots": g.Riots.values, "events_total": tot.values, "fatalities": g.FATALITIES.values,
            "prev4w": prev4.values, "trend_ratio": ratio.round(2).values,
            "next2w_events": next2.values, "real_escalation_label": lab.values}))
    df = pd.concat(out, ignore_index=True)
    df.to_csv(REGION_WEEKLY, index=False)
    print(df.shape, "->", REGION_WEEKLY)
    print(df.real_escalation_label.value_counts(dropna=False, normalize=True).round(2).to_dict())

if __name__ == "__main__": main()
