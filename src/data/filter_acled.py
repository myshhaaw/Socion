
import pandas as pd

# Read original ACLED Excel dataset
df = pd.read_excel("data/raw/ACLED/acled_original.xlsx")

print("Original rows:", len(df))
print("Columns:", df.columns.tolist())

# Filter only India
india_df = df[
    df["COUNTRY"].astype(str).str.strip().str.lower() == "india"
].copy()

# Convert WEEK into datetime
india_df["WEEK"] = pd.to_datetime(
    india_df["WEEK"], errors="coerce"
)

# Filter dates: 2018–2023
india_df = india_df[
    (india_df["WEEK"] >= "2018-01-01") &
    (india_df["WEEK"] < "2024-01-01")
]

# Save filtered dataset
india_df.to_csv(
    "data/raw/ACLED/acled_india.csv",
    index=False
)

print("\nIndia records:", len(india_df))
print("Date range:", india_df["WEEK"].min(), "to", india_df["WEEK"].max())
print("Filtered dataset saved successfully!")
