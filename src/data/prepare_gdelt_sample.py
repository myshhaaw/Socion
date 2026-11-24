"""GDELT -> clean, map state names to ACLED, drop duplicates/junk -> data/interim/gdelt_urls_sample.csv"""
import pandas as pd
from config import *

def main(per_year=100, seed=42):
    g = pd.read_csv(GDELT_CSV)
    g["date"] = pd.to_datetime(g.SQLDATE.astype(str), format="%Y%m%d")
    g["country"] = "India"
    g["state"] = g.ActionGeo_FullName.str.split(", ").str[-2].replace(STATE_ALIASES)  # "City, State, India"
    acled_states = set(pd.read_csv(REGION_WEEKLY).query("country=='India'").admin1)
    g["state_ok"] = g.state.isin(acled_states)
    print(f"state matched to ACLED: {g.state_ok.mean():.1%}")
    g = g[g.state_ok]
    g["domain"] = g.SOURCEURL.str.extract(r"https?://(?:www\.)?([^/]+)")[0]
    (g.domain.value_counts().rename_axis("domain").reset_index(name="n")).to_csv(REPORTS/"gdelt_domains.csv", index=False)
    g = g[~g.domain.isin(JUNK_DOMAINS)]
    g = g.sort_values("NumMentions", ascending=False).drop_duplicates("SOURCEURL")
    s = (g.groupby(g.date.dt.year, group_keys=False)
           .apply(lambda x: x.sample(min(per_year, len(x)), random_state=seed))).reset_index(drop=True)
    s.insert(0, "article_id", [f"G{i:04d}" for i in range(len(s))])
    s = s.rename(columns={"SOURCEURL": "url", "AvgTone": "avg_tone", "NumMentions": "num_mentions",
                          "GLOBALEVENTID": "gdelt_event_id"})
    s[["article_id","url","date","country","state","domain","gdelt_event_id","avg_tone","num_mentions"]] \
        .to_csv(INTERIM/"gdelt_urls_sample.csv", index=False)
    print(len(s), "urls sampled ->", INTERIM/"gdelt_urls_sample.csv"); print(s.date.dt.year.value_counts().sort_index().to_dict())

if __name__ == "__main__": main()
