import pandas as pd
from pathlib import Path

from score import score, explain

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "data" / "clustered.csv"
OUTPUT = ROOT / "data" / "scored.csv"

df = pd.read_csv(INPUT)

scored = score(df)

scored["priority_explanation"] = scored.apply(
    explain,
    axis=1
)

scored.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("Priority scoring complete.")
print("Saved to data/scored.csv")
print()

top = scored.sort_values(
    "priority_score",
    ascending=False
).head(10)

print(
    top[
        [
            "id",
            "category",
            "location_text",
            "priority_score"
        ]
    ].to_string(index=False)
)