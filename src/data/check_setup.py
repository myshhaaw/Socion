"""Health check for the data setup. Run from the repo root:
    python3 -m src.data.check_setup
"""
import importlib
import subprocess
import sys

import pandas as pd

from src.config import ROOT, RAW, INTERIM, PROCESSED, ANNOT, REPORTS

problems = 0

def say(ok, msg, level="fail"):
    global problems
    if ok:
        print(f"✅ {msg}")
    elif level == "warn":
        print(f"⚠️  {msg}")
    else:
        problems += 1
        print(f"❌ {msg}")

def git(*args):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()

ACLED_COLS = ["WEEK", "COUNTRY", "ADMIN1", "EVENT_TYPE", "SUB_EVENT_TYPE", "EVENTS", "FATALITIES"]
GDELT_COLS = ["GLOBALEVENTID", "SQLDATE", "EventCode", "EventRootCode", "NumMentions",
              "NumArticles", "AvgTone", "ActionGeo_FullName", "ActionGeo_CountryCode",
              "ActionGeo_Lat", "ActionGeo_Long", "SOURCEURL"]
CUT_FILES = ["keywords_english.txt", "known_annotations.csv",
             "labelled_tweets_is_general_unrest.csv",
             "labelled_tweets_is_protest_event.csv", "majority_annotation_results.csv"]

print("\n--- 1. Environment")
v = sys.version_info
say(v[:2] == (3, 13), f"Python {v.major}.{v.minor}.{v.micro} (team uses 3.13)", "warn")
say(sys.prefix != sys.base_prefix, "virtual environment is active")
for lib, required in [("pandas", True), ("numpy", True), ("sklearn", True),
                      ("matplotlib", True), ("seaborn", True), ("tabulate", False)]:
    try:
        importlib.import_module(lib)
        say(True, f"{lib} installed")
    except ImportError:
        say(False, f"{lib} missing -> pip install {'scikit-learn' if lib == 'sklearn' else lib}",
            "fail" if required else "warn")

print("\n--- 2. Git")
_, branch = git("rev-parse", "--abbrev-ref", "HEAD")
say(branch not in ("main", "master"), f"on branch '{branch}' (never work on main)")
for p in ["data/raw/ACLED/x.csv", "data/raw/GDELT/x.csv", "data/raw/CUT/x.csv",
          "data/interim/acled_cleaned/x.csv", ".env", ".venv/bin/python"]:
    code, _ = git("check-ignore", "-q", p)
    say(code == 0, f"git ignores {p}")
_, tracked = git("ls-files", "data/raw", "data/interim")
bad = [f for f in tracked.splitlines() if not f.endswith(".gitkeep")]
say(not bad, "no raw/interim data tracked by git" if not bad else f"TRACKED data files: {bad[:5]}")
_, files = git("ls-files")
big = [f for f in files.splitlines() if (ROOT / f).exists() and (ROOT / f).stat().st_size > 5_000_000]
say(not big, "no tracked file over 5 MB" if not big else f"big tracked files: {big}")

print("\n--- 3. Folders")
for d in [RAW / "CUT", RAW / "ACLED", RAW / "GDELT", INTERIM, PROCESSED, ANNOT, REPORTS,
          ROOT / "notebooks", ROOT / "src" / "data"]:
    say(d.exists(), str(d.relative_to(ROOT)))

print("\n--- 4. Raw data files")
def check_csv_source(name, required):
    folder = RAW / name
    files = sorted(folder.glob("*.csv")) if folder.exists() else []
    other = [f.name for f in folder.iterdir()
             if f.suffix.lower() in {".xlsx", ".xls", ".zip", ".json", ".parquet"}] if folder.exists() else []
    if not files:
        say(False, f"{name}: no CSV in data/raw/{name}/" + (f" (found {other}; convert to CSV)" if other else ""))
        return
    for f in files:
        cols = pd.read_csv(f, nrows=0).columns.tolist()
        n = len(pd.read_csv(f, usecols=[0], low_memory=False))
        missing = [c for c in required if c not in cols]
        say(not missing, f"{name}/{f.name}: {n:,} rows, {len(cols)} columns"
            + (f" | MISSING: {missing}" if missing else ""))
        if missing:
            print("   columns found:", cols)

check_csv_source("ACLED", ACLED_COLS)
check_csv_source("GDELT", GDELT_COLS)
for f in CUT_FILES:
    say((RAW / "CUT" / f).exists(), f"CUT/{f}")

print("\n--- 5. Documentation")
readme = ROOT / "data" / "README.md"
if readme.exists():
    t = readme.read_text()
    say(all(k in t for k in ["CUT", "ACLED", "GDELT"]), "data/README.md mentions all three sources")
    say("<" not in t, "data/README.md has no <placeholders> left", "warn")
else:
    say(False, "data/README.md missing")

print(f"\n{'All good ✅' if problems == 0 else f'{problems} problem(s) ❌ - fix these first'}")
