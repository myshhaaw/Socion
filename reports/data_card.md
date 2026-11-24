# Data card

| Dataset | File | Rows | Real/Synthetic | Use |
|---|---|---|---|---|
| ACLED weekly (Asia-Pacific) | data/raw/acled_asia_pacific_weekly.xlsx | ~224k | Real | Region context + real escalation label |
| GDELT India protest events | data/raw/gdelt_india_protest_2018_2023.csv | ~273k | Real (noisy) | Source of real article URLs |
| Synthetic news articles | data/raw/synthetic_civil_unrest_news.csv | 1,450 | Synthetic text, real places/context | Training + development |
| Region weekly table | data/processed/region_weekly.csv | ~304k | Derived from ACLED | API context lookups |
| Real test set | data/processed/real_test_set.csv | ~200 | Real text, hand-labelled | Final evaluation only |

## Known limitations
- Synthetic text is template-based: scores on it overstate real performance. Report real-test-set results separately.
- Synthetic `escalation_risk` is a rule-plus-noise label. Real articles use `real_escalation_label` from ACLED (next 2 weeks).
- GDELT coding is noisy and covers India only; many article URLs are dead or paywalled (see reports/scrape_report.csv).
- ACLED data is licensed: do not publish raw ACLED files publicly; keep DVC remote private.

## Label definitions
- Real escalation: HIGH if next-2-week protest+riot events >= max(5, 1.5 x baseline) or any fatalities; MEDIUM if >= max(2, 1.1 x baseline); else LOW
  (baseline = mean weekly events over the previous 52 weeks x 2). Thresholds live in src/data/config.py.
- Time split: train 2018-2021, val 2022, test 2023.
