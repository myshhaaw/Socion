# Data Quality Report

- ACLED weekly rows: 24,350 | states: 36 | weeks: 313 (2018-01-06 to 2023-12-30)
- GDELT records: 171,788 | missing state: 8.4%
- Registry rows: 196,138 (171,788 GDELT events)
- GDELT link status (state-week level):
  - linked_state_week: 88.3%
  - no_state: 8.4%
  - no_acled_match: 3.3%
- Forecasting rows: 10,728 | positive rate: 79.3%

## Rows and positive rate per split

| split   |   size |     mean |
|:--------|-------:|---------:|
| test    |   1836 | 0.817538 |
| train   |   7056 | 0.783163 |
| val     |   1836 | 0.805011 |

## Checks passed
- unique event IDs; no duplicate (state, week) rows
- target and 4-week feature recomputed independently on 300 random rows
- train < val < test; no target window crosses a split boundary

## Limitations
- ACLED file is aggregated: linking is state-week level, not event-level.
- GDELT reflects news coverage, not necessarily real events.