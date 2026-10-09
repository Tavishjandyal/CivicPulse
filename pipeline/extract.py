from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import ollama
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

MODEL_NAME = "qwen2.5:7b-instruct"

CATEGORIES = [
    "road_damage",
    "streetlight",
    "drainage",
    "water_leak",
    "waste",
    "other",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_FILE = DATA_DIR / "extract_cache.json"

DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# EXPECTED OUTPUT FORMAT
# ---------------------------------------------------------

class ExtractionResult(BaseModel):
    category: Literal[
        "road_damage",
        "streetlight",
        "drainage",
        "water_leak",
        "waste",
        "other",
    ]

    subcategory: str = Field(min_length=1, max_length=50)

    location_text: str | None = None

    duration_days: int | None = Field(default=None, ge=0)

    severity: int = Field(ge=1, le=10)

    infrastructure: str = Field(min_length=1, max_length=80)


# ---------------------------------------------------------
# PROMPT GIVEN TO QWEN
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You extract structured information from civic complaints from Jaipur, India.

Complaints may be written in:
- English
- Hindi in Devanagari
- Hinglish in Roman script
- informal language with spelling mistakes

You MUST return valid JSON only.

Allowed categories:

road_damage:
potholes, broken roads, damaged pavement, unsafe road surface.

streetlight:
streetlight not working, dim lights, damaged street poles or public lighting.

drainage:
blocked drains, overflowing drains, sewage overflow, open drains.

water_leak:
burst water pipe, leaking municipal pipe, water supply leakage.

waste:
garbage piles, missed garbage collection, overflowing public bins.

other:
anything that does not clearly fit the categories above.

Severity rules:

1-3 = minor inconvenience.
4-6 = moderate problem.
7-8 = major disruption or meaningful safety concern.
9-10 = immediate danger, such as open manhole, exposed live wire,
major road hazard, or severe dangerous flooding.

Duration:
Convert duration to number of days.

Examples:
"3 din se" = 3
"teen din se" = 3
"do hafte se" = 14
"2 weeks" = 14
"kal se" = 1
"aaj se" = 0

If no duration is mentioned, use null.

location_text:
Preserve the location approximately as the citizen wrote it.
If there is no location, use null.

subcategory:
Give a short description, around 1-3 words.

infrastructure:
Give a short noun phrase describing the affected infrastructure.

Return ONLY this JSON structure:

{
  "category": "...",
  "subcategory": "...",
  "location_text": "... or null",
  "duration_days": 0,
  "severity": 1,
  "infrastructure": "..."
}

Examples:

Complaint:
"Huge pothole on Tonk Road near the signal, 2 weeks now, bike nearly skidded"

Output:
{
  "category": "road_damage",
  "subcategory": "large pothole",
  "location_text": "Tonk Road near the signal",
  "duration_days": 14,
  "severity": 8,
  "infrastructure": "road surface"
}

Complaint:
"Bhai Mansarovar mein nali overflow ho rahi hai 3 din se, badbu aa rahi hai"

Output:
{
  "category": "drainage",
  "subcategory": "drain overflow",
  "location_text": "Mansarovar",
  "duration_days": 3,
  "severity": 6,
  "infrastructure": "drain"
}

Complaint:
"राजा पार्क में तीन दिन से स्ट्रीट लाइट बंद है"

Output:
{
  "category": "streetlight",
  "subcategory": "light failure",
  "location_text": "राजा पार्क",
  "duration_days": 3,
  "severity": 4,
  "infrastructure": "streetlight"
}

Complaint:
"Vaishali nagar me pipe foot gya kal se pani road pe beh rha"

Output:
{
  "category": "water_leak",
  "subcategory": "burst pipe",
  "location_text": "Vaishali nagar",
  "duration_days": 1,
  "severity": 7,
  "infrastructure": "water pipe"
}

Complaint:
"कचरे का ढेर गौरव टावर के पास कई दिनों से पड़ा है"

Output:
{
  "category": "waste",
  "subcategory": "garbage pile",
  "location_text": "गौरव टावर के पास",
  "duration_days": null,
  "severity": 5,
  "infrastructure": "garbage"
}

Complaint:
"Jhotwara mein khula manhole hai koi gir skta h"

Output:
{
  "category": "drainage",
  "subcategory": "open manhole",
  "location_text": "Jhotwara",
  "duration_days": null,
  "severity": 10,
  "infrastructure": "manhole"
}
"""


# ---------------------------------------------------------
# CACHE
# ---------------------------------------------------------

def load_cache() -> dict:
    if not CACHE_FILE.exists():
        return {}

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------
# SINGLE COMPLAINT EXTRACTION
# ---------------------------------------------------------

def extract(text: str) -> dict:
    """
    Convert one civic complaint into structured information.
    """

    if not isinstance(text, str) or not text.strip():
        return {
            "category": "other",
            "subcategory": "unknown",
            "location_text": None,
            "duration_days": None,
            "severity": 3,
            "infrastructure": "unknown",
        }

    last_error = None

    # First attempt + two retries
    for _ in range(3):

        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": text,
                    },
                ],
                format="json",
                options={
                    "temperature": 0,
                },
            )

            content = response["message"]["content"]

            data = json.loads(content)

            validated = ExtractionResult.model_validate(data)

            return validated.model_dump()

        except (json.JSONDecodeError, ValidationError, KeyError, Exception) as e:
            last_error = e

    print(f"Extraction failed: {last_error}")

    # Safe fallback so one bad complaint never crashes everything
    return {
        "category": "other",
        "subcategory": "unknown",
        "location_text": None,
        "duration_days": None,
        "severity": 3,
        "infrastructure": "unknown",
    }


# ---------------------------------------------------------
# BATCH EXTRACTION
# ---------------------------------------------------------

def extract_batch(df: pd.DataFrame, workers: int = 4) -> pd.DataFrame:

    if "raw_text" not in df.columns:
        raise ValueError("DataFrame must contain a 'raw_text' column.")

    cache = load_cache()

    results = [None] * len(df)

    def process_row(index: int, text: str):

        key = text_hash(text)

        if key in cache:
            return index, cache[key]

        result = extract(text)

        return index, result

    with ThreadPoolExecutor(max_workers=workers) as executor:

        futures = {
            executor.submit(
                process_row,
                i,
                str(row["raw_text"])
            ): i
            for i, (_, row) in enumerate(df.iterrows())
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Extracting complaints"
        ):

            index, result = future.result()

            results[index] = result

            text = str(df.iloc[index]["raw_text"])

            cache[text_hash(text)] = result

    save_cache(cache)

    result_df = df.copy()

    result_df["category"] = [r["category"] for r in results]
    result_df["subcategory"] = [r["subcategory"] for r in results]
    result_df["location_text"] = [r["location_text"] for r in results]

    # P2 will fill these later
    result_df["lat"] = None
    result_df["lon"] = None

    result_df["duration_days"] = [r["duration_days"] for r in results]
    result_df["severity"] = [r["severity"] for r in results]
    result_df["infrastructure"] = [r["infrastructure"] for r in results]

    output_file = DATA_DIR / "extracted.csv"

    result_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nSaved extracted data to: {output_file}")

    return result_df


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def evaluate(df: pd.DataFrame) -> dict:

    metrics = {}

    if "true_category" in df.columns:

        category_accuracy = (
            df["category"] == df["true_category"]
        ).mean()

        metrics["category_accuracy"] = round(
            category_accuracy * 100,
            2
        )

    if "true_severity" in df.columns:

        valid = df[
            df["true_severity"].notna()
            & df["severity"].notna()
        ]

        if len(valid) > 0:

            severity_correct = (
                (
                    valid["severity"].astype(float)
                    - valid["true_severity"].astype(float)
                ).abs()
                <= 2
            ).mean()

            metrics["severity_within_2"] = round(
                severity_correct * 100,
                2
            )

    if "true_duration_days" in df.columns:

        valid = df[
            df["true_duration_days"].notna()
            & df["duration_days"].notna()
        ]

        if len(valid) > 0:

            duration_correct = (
                valid["duration_days"].astype(float)
                == valid["true_duration_days"].astype(float)
            ).mean()

            metrics["duration_accuracy"] = round(
                duration_correct * 100,
                2
            )

    print("\nEvaluation results")

    for key, value in metrics.items():
        print(f"{key}: {value}%")

    return metrics


# ---------------------------------------------------------
# SIMPLE TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    input_file = DATA_DIR / "complaints_raw.csv"

    df = pd.read_csv(input_file)

    result_df = extract_batch(df, workers=4)

    evaluate(result_df)