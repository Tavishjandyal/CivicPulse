from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ollama
import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm


# ---------------------------------------------------------
# FILES
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data" / "extracted.csv"
OUTPUT_FILE = ROOT / "data" / "clustered.csv"
CACHE_FILE = ROOT / "data" / "dedupe_fingerprints.json"

MODEL = "qwen2.5:7b-instruct"


# ---------------------------------------------------------
# QWEN PROMPT
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You normalize civic complaints from Jaipur for duplicate detection.

The complaint may be English, Hindi or Hinglish.

Return JSON ONLY with:

{
  "issue_key": "...",
  "location_key": "..."
}

issue_key:
Create a very short canonical English snake_case label describing the
actual physical civic issue.

Examples:

"nali overflow"
"sewer ka pani jama"
"सीवर का पानी बह रहा"
-> drain_overflow

"big pothole"
"bada gaddha"
"बड़ा गड्ढा"
-> pothole

"street light band"
"streetlight not working"
-> streetlight_out

"khula manhole"
"open manhole"
-> open_manhole

"garbage pile"
"kachra pada hai"
-> garbage_accumulation

"garbage not collected"
-> missed_garbage_collection

"pipe leaking"
-> water_pipe_leak

"pipe burst"
-> burst_water_pipe

"no water supply"
-> no_water_supply

"low water pressure"
-> low_water_pressure

"traffic signal not working"
-> traffic_signal_failure

"tree branch fallen"
-> fallen_tree_branch

"hanging tree branch"
-> hanging_tree_branch

"electrical box open"
-> open_electrical_box

"wire lying exposed"
-> exposed_electrical_wire


location_key:
Translate/normalize the location into short Roman-English form.

Examples:

"मानसरोवर सेक्टर 7"
"Mansarovar sec 7"
"Sector 7 Mansarovar"
-> "mansarovar sector 7"

"गौरव टावर के पास साइड वाली गली"
"GT ke side wali gali"
"side lane near Gaurav Tower"
-> "gaurav tower side lane"

"राजा पार्क मार्केट के पास"
"Raja Park market ke paas"
-> "raja park market"

Remove words like:
near, paas, ke paas, outside, bahar, mein, me, at.

Keep important landmarks, sector numbers, road names, crossings and lane numbers.

Do not invent a location.

If no useful location exists, use "".
"""


# ---------------------------------------------------------
# CACHE
# ---------------------------------------------------------

def load_cache():
    if not CACHE_FILE.exists():
        return {}

    try:
        with CACHE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(cache):
    with CACHE_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            cache,
            f,
            ensure_ascii=False,
            indent=2
        )


def hash_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------
# CREATE ONE FINGERPRINT
# ---------------------------------------------------------

def fingerprint(row):
    raw_text = str(row.get("raw_text", ""))

    prompt = f"""
Complaint:
{raw_text}

Existing extraction:
category: {row.get("category", "")}
subcategory: {row.get("subcategory", "")}
location: {row.get("location_text", "")}
infrastructure: {row.get("infrastructure", "")}
"""

    for _ in range(3):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                format="json",
                options={
                    "temperature": 0
                }
            )

            result = json.loads(
                response["message"]["content"]
            )

            return {
                "issue_key": str(
                    result.get("issue_key", "")
                ).strip().lower(),

                "location_key": str(
                    result.get("location_key", "")
                ).strip().lower(),
            }

        except Exception:
            pass

    return {
        "issue_key": str(
            row.get("subcategory", "")
        ).lower(),

        "location_key": str(
            row.get("location_text", "")
        ).lower(),
    }


# ---------------------------------------------------------
# CREATE FINGERPRINTS FOR ALL ROWS
# ---------------------------------------------------------

def build_fingerprints(df, workers=4):

    cache = load_cache()

    fingerprints = [None] * len(df)

    def process(i):

        text = str(df.iloc[i]["raw_text"])
        key = hash_text(text)

        if key in cache:
            return i, cache[key]

        fp = fingerprint(
            df.iloc[i]
        )

        return i, fp

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = [
            executor.submit(process, i)
            for i in range(len(df))
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Creating fingerprints"
        ):

            i, fp = future.result()

            fingerprints[i] = fp

            text = str(
                df.iloc[i]["raw_text"]
            )

            cache[
                hash_text(text)
            ] = fp

    save_cache(cache)

    return fingerprints


# ---------------------------------------------------------
# SIMILARITY
# ---------------------------------------------------------

def location_similarity(a, b):

    a = str(a or "").strip().lower()
    b = str(b or "").strip().lower()

    if not a or not b:
        return 0.0

    return (
        fuzz.token_set_ratio(a, b)
        / 100.0
    )


def issue_similarity(a, b):

    a = str(a or "").strip().lower()
    b = str(b or "").strip().lower()

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    return (
        fuzz.token_set_ratio(
            a.replace("_", " "),
            b.replace("_", " ")
        )
        / 100.0
    )


# ---------------------------------------------------------
# DUPLICATE DECISION
# ---------------------------------------------------------

def duplicate_score(
    row1,
    row2,
    fp1,
    fp2
):

    issue = issue_similarity(
        fp1["issue_key"],
        fp2["issue_key"]
    )

    location = location_similarity(
        fp1["location_key"],
        fp2["location_key"]
    )

    # Duration can support a duplicate match.
    d1 = row1.get("duration_days")
    d2 = row2.get("duration_days")

    duration_bonus = 0.0

    if (
        pd.notna(d1)
        and pd.notna(d2)
    ):

        try:
            difference = abs(
                float(d1) - float(d2)
            )

            if difference == 0:
                duration_bonus = 0.05

            elif difference <= 1:
                duration_bonus = 0.02

        except Exception:
            pass

    score = (
        issue * 0.60
        + location * 0.35
        + duration_bonus
    )

    return min(score, 1.0)




def is_duplicate(
    row1,
    row2,
    fp1,
    fp2
):

    issue = issue_similarity(
        fp1["issue_key"],
        fp2["issue_key"]
    )

    location = location_similarity(
        fp1["location_key"],
        fp2["location_key"]
    )

    loc1 = str(fp1["location_key"] or "").lower()
    loc2 = str(fp2["location_key"] or "").lower()

    import re

    numbers1 = set(re.findall(r"\d+", loc1))
    numbers2 = set(re.findall(r"\d+", loc2))

    # sector 7 should not match sector 9
    if numbers1 and numbers2:
        if numbers1 != numbers2:
            return False, 0.0

    # issue must be strongly similar
    if issue < 0.95:
        return False, 0.0

    # location must also match well
    if location < 0.92:
        return False, 0.0

    generic_words = {
        "mansarovar",
        "malviya nagar",
        "vaishali nagar",
        "jagatpura",
        "raja park",
        "jhotwara",
        "sanganer",
        "bani park",
    }

    # vague locality names require almost exact match
    if loc1 in generic_words or loc2 in generic_words:
        if location < 0.95:
            return False, 0.0

    score = duplicate_score(
        row1,
        row2,
        fp1,
        fp2
    )

    return True, score
def assign_clusters(df):
    ...
def assign_clusters(df):

    print()
    print("Creating AI fingerprints...")

    fingerprints = build_fingerprints(
        df,
        workers=4
    )

    # Save fingerprints so we can inspect/demo them.
    result = df.copy()

    result["issue_key"] = [
        fp["issue_key"]
        for fp in fingerprints
    ]

    result["location_key"] = [
        fp["location_key"]
        for fp in fingerprints
    ]

    clusters = []

    print()
    print("Grouping duplicate complaints...")

    for i in range(len(df)):

        placed = False

        for cluster in clusters:

            # Compare against EVERY existing cluster member.
            matches = []

            for member in cluster:

                dup, score = is_duplicate(
                    df.iloc[i],
                    df.iloc[member],
                    fingerprints[i],
                    fingerprints[member]
                )

                matches.append(dup)

            # Prevent chain-merging.
            if matches and all(matches):

                cluster.append(i)
                placed = True
                break

        if not placed:
            clusters.append([i])

    cluster_ids = [None] * len(df)

    for number, members in enumerate(
        clusters,
        start=1
    ):

        cid = f"cluster_{number:03d}"

        for member in members:
            cluster_ids[member] = cid

    result["cluster_id"] = (
        cluster_ids
    )

    size_map = (
        result["cluster_id"]
        .value_counts()
    )

    result["cluster_size"] = (
        result["cluster_id"]
        .map(size_map)
    )

    return result


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def report(df):

    tp = 0
    fp = 0
    fn = 0

    for i in range(len(df)):

        for j in range(
            i + 1,
            len(df)
        ):

            predicted_same = (
                df.iloc[i]["cluster_id"]
                ==
                df.iloc[j]["cluster_id"]
            )

            actual_same = (
                str(
                    df.iloc[i][
                        "true_cluster_id"
                    ]
                )
                ==
                str(
                    df.iloc[j][
                        "true_cluster_id"
                    ]
                )
            )

            if (
                predicted_same
                and actual_same
            ):
                tp += 1

            elif (
                predicted_same
                and not actual_same
            ):
                fp += 1

            elif (
                not predicted_same
                and actual_same
            ):
                fn += 1

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0
    )

    print()
    print(
        "=============================="
    )
    print(
        "DUPLICATE DETECTION METRICS"
    )
    print(
        "=============================="
    )

    print(
        f"Precision: {precision:.3f}"
    )

    print(
        f"Recall:    {recall:.3f}"
    )

    print(
        f"F1:        {f1:.3f}"
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ---------------------------------------------------------
# LIVE MATCH
# ---------------------------------------------------------

def match_existing(
    new_row: dict,
    df: pd.DataFrame
):

    new_series = pd.Series(
        new_row
    )

    new_fp = fingerprint(
        new_series
    )

    best_score = 0.0
    best_row = None

    for _, row in df.iterrows():

        old_fp = {
            "issue_key": row.get(
                "issue_key",
                row.get(
                    "subcategory",
                    ""
                )
            ),

            "location_key": row.get(
                "location_key",
                row.get(
                    "location_text",
                    ""
                )
            ),
        }

        duplicate, score = is_duplicate(
            new_series,
            row,
            new_fp,
            old_fp
        )

        if (
            duplicate
            and score > best_score
        ):
            best_score = score
            best_row = row

    if best_row is None:

        return {
            "cluster_id": None,
            "is_duplicate": False,
            "best_match_id": None,
            "score": round(
                best_score,
                3
            )
        }

    return {
        "cluster_id": best_row.get(
            "cluster_id"
        ),

        "is_duplicate": True,

        "best_match_id": best_row.get(
            "id"
        ),

        "score": round(
            best_score,
            3
        )
    }


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        "Reading extracted complaints..."
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Found {len(df)} complaints."
    )

    clustered = assign_clusters(
        df
    )

    clustered.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        "Saved:",
        OUTPUT_FILE
    )

    print(
        "Total clusters:",
        clustered[
            "cluster_id"
        ].nunique()
    )

    report(
        clustered
    )