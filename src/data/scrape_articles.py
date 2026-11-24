"""Scrape article text for sampled GDELT URLs (resumable).  Pilot first:  python src/data/scrape_articles.py --n 50"""
import argparse, json, time, pandas as pd
from tqdm import tqdm
from trafilatura import fetch_url, extract
from config import *

def main(n, delay):
    urls = pd.read_csv(INTERIM/"gdelt_urls_sample.csv").head(n)
    done = {json.loads(l)["article_id"] for l in open(SCRAPED_JSONL)} if SCRAPED_JSONL.exists() else set()
    with open(SCRAPED_JSONL, "a") as f:
        for r in tqdm(urls.itertuples(), total=len(urls)):
            if r.article_id in done: continue
            try:
                html = fetch_url(r.url)
                text = extract(html, include_comments=False, include_tables=False) if html else None
            except Exception:
                text = None
            text = text or ""
            f.write(json.dumps({"article_id": r.article_id, "url": r.url, "domain": r.domain,
                                "ok": len(text) > 300, "chars": len(text), "text": text}) + "\n"); f.flush()
            time.sleep(delay)
    s = pd.read_json(SCRAPED_JSONL, lines=True)
    print(f"success rate: {s.ok.mean():.1%} of {len(s)}")
    s.groupby("domain").ok.agg(["mean", "count"]).sort_values("count", ascending=False).to_csv(REPORTS/"scrape_report.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=600); ap.add_argument("--delay", type=float, default=1.0)
    a = ap.parse_args(); main(a.n, a.delay)
