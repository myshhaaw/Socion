import re
import pandas as pd
from src.config import INTERIM

STATE_ALIASES = {  # EXTEND after exploring your data
    "orissa": "odisha", "uttaranchal": "uttarakhand", "pondicherry": "puducherry",
    "nct of delhi": "delhi", "new delhi": "delhi",
    "jammu and kashmir": "jammu & kashmir",
}

def norm_place(x):
    if pd.isna(x):
        return None
    x = re.sub(r"[^a-z0-9 &]", " ", str(x).lower())
    x = re.sub(r"\s+", " ", x).strip()
    return x or None

def norm_state(x):
    x = norm_place(x)
    return STATE_ALIASES.get(x, x)

def clean_text(t):
    t = re.sub(r"http\S+", "<URL>", str(t))
    t = re.sub(r"@\w+", "@USER", t)
    return re.sub(r"\s+", " ", t).strip()

def category_family(cat):
    """strike is treated as a kind of protest for linking."""
    return "riot" if cat == "riot" else "protest"

def save_interim(df, folder, name):
    out = INTERIM / folder
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / name, index=False)
    print(f"saved {len(df):,} rows -> {out / name}")
OFFICIAL_STATES = {norm_state(s) for s in [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal", "Andaman and Nicobar Islands", "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir", "Ladakh",
    "Lakshadweep", "Puducherry"]}
