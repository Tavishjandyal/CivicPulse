import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "data" / "extracted.csv"

df = pd.read_csv(FILE)

wrong = df[df["category"] != df["true_category"]]

print("Wrong category predictions:", len(wrong))
print()

for _, row in wrong.iterrows():
    print("--------------------------------")
    print("Complaint:", row["raw_text"])
    print("Correct:", row["true_category"])
    print("AI said:", row["category"])
    print()

wrong.to_csv(
    ROOT / "data" / "category_errors.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Saved mistakes to data/category_errors.csv")