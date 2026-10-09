import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "data" / "scored.csv"

df = pd.read_csv(FILE)

# Demo ward assignment based on locality text
def assign_demo_ward(location):
    text = str(location).lower()

    if "mansarovar" in text:
        return "Ward 63"

    if "malviya" in text:
        return "Ward 127"

    if "vaishali" in text:
        return "Ward 70"

    if "jagatpura" in text:
        return "Ward 119"

    if "raja park" in text:
        return "Ward 93"

    if "jhotwara" in text:
        return "Ward 45"

    if "sanganer" in text:
        return "Ward 106"

    return "Ward Other"


df["ward"] = df["location_text"].apply(assign_demo_ward)


# Demo status
def assign_status(score):
    score = float(score)

    if score >= 70:
        return "Open"

    if score >= 50:
        return "In Progress"

    return "Resolved"


df["status"] = df["priority_score"].apply(assign_status)


# Demo assignment time
now = datetime.now()

def assign_time(score):
    score = float(score)

    if score >= 70:
        return now - timedelta(hours=30)

    if score >= 50:
        return now - timedelta(hours=18)

    return now - timedelta(hours=8)


df["assigned_at"] = df["priority_score"].apply(assign_time)


df.to_csv(
    FILE,
    index=False,
    encoding="utf-8-sig"
)

print("Accountability demo fields added.")
print()
print(df[[
    "id",
    "ward",
    "status",
    "assigned_at",
    "priority_score"
]].head(10).to_string(index=False))