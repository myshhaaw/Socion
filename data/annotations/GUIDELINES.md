# Annotation guidelines (real test set)

Label each article from `text_excerpt` (open `url` if you need more). One row = one article.

**is_unrest_event** = 1 only if the article mainly reports a *specific* protest, riot, strike, march or blockade that
happened, is happening, or is announced for a specific time. Otherwise 0.
Label 0 for: opinion/analysis about unrest in general, history/anniversaries, election campaign rallies, sports,
police/crime news with no public gathering, passing mention of a protest as background, diplomatic "protest notes".

**event_type** (if 1): protest (gathering/sit-in/demonstration), riot (violent, property damage, clashes),
strike (work stoppage, bandh), march (moving rally/procession), blockade (roads/rail blocked). If mixed, pick the main one
(violence -> riot). If 0, write `none`.

**event_status** (if 1): past / ongoing / planned. If 0, write `none`.

**state_in_text**: the Indian state the main event happened in, as written in the article (blank if not stated).
**cause**: the demand/reason, copied as a short phrase from the article (blank if not stated).
**notes**: anything unclear. When unsure, make your best choice and write why.

Do not edit other columns. Save your finished sheet as `labeled_<yourname>.csv` in this folder.
The first 20 rows are shared by everyone: label them independently, do not discuss until all are done.
