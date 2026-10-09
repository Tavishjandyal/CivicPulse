import pandas as pd
from pathlib import Path

from geocode import geocode

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "data" / "extracted.csv"

df = pd.read_csv(FILE)

for i, row in df.iterrows():
    location = row.get("location_text")

    if pd.isna(location) or not str(location).strip():
        continue

    lat, lon, matched_name, confidence = geocode(str(location))

    df.at[i, "lat"] = lat
    df.at[i, "lon"] = lon

    print(
        f"{i + 1}/{len(df)} | "
        f"{location} -> {matched_name} "
        f"({lat}, {lon}) confidence={confidence}"
    )

df.to_csv(FILE, index=False, encoding="utf-8-sig")

print()
print("Geocoding complete.")
print("Updated data/extracted.csv")