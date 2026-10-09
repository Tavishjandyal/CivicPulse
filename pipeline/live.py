from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from pipeline.extract import extract
from pipeline.geocode import geocode
from pipeline.dedupe import match_existing
from pipeline.score import score, explain
from pipeline.analytics import route


ROOT = Path(__file__).resolve().parent.parent
CLUSTERED_FILE = ROOT / "data" / "clustered.csv"


def load_existing():

    if CLUSTERED_FILE.exists():
        return pd.read_csv(CLUSTERED_FILE)

    return pd.DataFrame()


def process_one(text: str) -> dict:

    # 1. Extract complaint information
    extracted = extract(text)

    category = extracted["category"]
    subcategory = extracted["subcategory"]
    location_text = extracted["location_text"]
    duration_days = extracted["duration_days"]
    severity = extracted["severity"]
    infrastructure = extracted["infrastructure"]


    # 2. Geocode location
    lat = None
    lon = None
    matched_location = None
    geo_confidence = 0.0

    if location_text:

        (
            lat,
            lon,
            matched_location,
            geo_confidence
        ) = geocode(location_text)


    complaint_id = "LIVE-" + datetime.now().strftime("%H%M%S")


    new_row = {
        "id": complaint_id,
        "raw_text": text,
        "language": "auto",
        "timestamp": datetime.now().isoformat(),

        "category": category,
        "subcategory": subcategory,
        "location_text": location_text,

        "lat": lat,
        "lon": lon,

        "duration_days": duration_days,
        "severity": severity,
        "infrastructure": infrastructure,
    }


    # 3. Duplicate detection
    existing = load_existing()

    duplicate_result = {
        "cluster_id": None,
        "is_duplicate": False,
        "best_match_id": None,
        "score": 0.0,
    }

    if not existing.empty:

        duplicate_result = match_existing(
            new_row,
            existing
        )


    # 4. Cluster assignment
    if duplicate_result["is_duplicate"]:

        cluster_id = duplicate_result["cluster_id"]

    else:

        cluster_id = "live_" + datetime.now().strftime("%H%M%S")


    new_row["cluster_id"] = cluster_id


    # 5. Priority scoring
    if not existing.empty:

        scoring_df = pd.concat(
            [
                existing,
                pd.DataFrame([new_row])
            ],
            ignore_index=True
        )

    else:

        scoring_df = pd.DataFrame([new_row])


    scored_df = score(scoring_df)

    scored_row = scored_df.iloc[-1]

    priority_score = float(
        scored_row["priority_score"]
    )

    priority_explanation = explain(
        scored_row
    )


    # 6. Department routing
    route_result = route(
        category,
        subcategory
    )

    department = route_result["department"]
    routing_reason = route_result["reason"]


    # 7. Final result
    return {

        "input": {
            "text": text
        },

        "extraction": {
            "category": category,
            "subcategory": subcategory,
            "location_text": location_text,
            "duration_days": duration_days,
            "severity": severity,
            "infrastructure": infrastructure,
        },

        "geocoding": {
            "lat": lat,
            "lon": lon,
            "matched_location": matched_location,
            "confidence": geo_confidence,
        },

        "duplicate_detection": duplicate_result,

        "priority": {
            "score": round(priority_score, 2),
            "explanation": priority_explanation,
        },

        "routing": {
            "department": department,
            "reason": routing_reason,
        },

        "complaint_id": complaint_id,

        "cluster_id": cluster_id,
    }


if __name__ == "__main__":

    test_text = (
        "Bhai Mansarovar sector 7 me "
        "nali overflow ho rahi hai "
        "3 din se, bahut badbu aa rahi hai"
    )

    result = process_one(test_text)

    print()
    print("======================================")
    print("CIVICPULSE LIVE PIPELINE")
    print("======================================")

    for section, value in result.items():

        print(f"\n--- {section.upper()} ---")
        print(value)