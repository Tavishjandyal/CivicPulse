import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "data" / "clustered.csv"

df = pd.read_csv(FILE)

false_pairs = []

for i in range(len(df)):
    for j in range(i + 1, len(df)):

        predicted_same = (
            str(df.iloc[i]["cluster_id"])
            ==
            str(df.iloc[j]["cluster_id"])
        )

        actually_same = (
            str(df.iloc[i]["true_cluster_id"])
            ==
            str(df.iloc[j]["true_cluster_id"])
        )

        if predicted_same and not actually_same:
            false_pairs.append({
                "id_1": df.iloc[i]["id"],
                "text_1": df.iloc[i]["raw_text"],
                "issue_key_1": df.iloc[i].get("issue_key", ""),
                "location_key_1": df.iloc[i].get("location_key", ""),

                "id_2": df.iloc[j]["id"],
                "text_2": df.iloc[j]["raw_text"],
                "issue_key_2": df.iloc[j].get("issue_key", ""),
                "location_key_2": df.iloc[j].get("location_key", ""),
            })

errors = pd.DataFrame(false_pairs)

output = ROOT / "data" / "dedupe_false_positives.csv"

errors.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("False duplicate pairs:", len(errors))
print("Saved:", output)

print()

for _, row in errors.head(15).iterrows():

    print("--------------------------------")

    print("Complaint 1:")
    print(row["text_1"])

    print("Issue:", row["issue_key_1"])
    print("Location:", row["location_key_1"])

    print()

    print("Complaint 2:")
    print(row["text_2"])

    print("Issue:", row["issue_key_2"])
    print("Location:", row["location_key_2"])