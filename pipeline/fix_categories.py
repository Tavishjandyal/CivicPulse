import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FILE = PROJECT_ROOT / "data" / "complaints_raw.csv"

df = pd.read_csv(FILE)

mapping = {
    "pothole": "road_damage",
    "footpath": "road_damage",
    "manhole": "drainage",
    "waterlogging": "drainage",
    "water_supply": "other",
    "electrical_hazard": "other",
    "tree_hazard": "other",
    "traffic_signal": "other",
    "stray_animals": "other",
    "park_maintenance": "other",
}

df["true_category"] = df["true_category"].replace(mapping)

df.to_csv(FILE, index=False, encoding="utf-8-sig")

print("Categories fixed successfully.")
print()
print(df["true_category"].value_counts())