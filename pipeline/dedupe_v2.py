from __future__ import annotations

from pathlib import Path
import math
import re
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# FILES / SETTINGS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data" / "extracted.csv"
OUTPUT_FILE = ROOT / "data" / "clustered.csv"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Starting threshold.
# tune() will test other values too.
DEFAULT_THRESHOLD = 0.78


# ---------------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------------

print("Loading duplicate-detection model...")

model = SentenceTransformer(MODEL_NAME)


# ---------------------------------------------------------
# TEXT HELPERS
# ---------------------------------------------------------

def clean_text(value) -> str:
    """Clean text while preserving Hindi/Devanagari."""

    if pd.isna(value):
        return ""

    text = unicodedata.normalize(
        "NFKC",
        str(value)
    ).lower().strip()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def make_semantic_text(row: pd.Series) -> str:
    """
    Create a cleaner description for the embedding model.

    We combine the original complaint with Qwen's extracted
    subcategory and infrastructure.
    """

    raw = clean_text(row.get("raw_text", ""))
    subcategory = clean_text(row.get("subcategory", ""))
    infrastructure = clean_text(row.get("infrastructure", ""))

    return (
        f"{raw}. "
        f"issue: {subcategory}. "
        f"infrastructure: {infrastructure}"
    )


# ---------------------------------------------------------
# LOCATION COMPARISON
# ---------------------------------------------------------

def location_similarity(row1: pd.Series, row2: pd.Series) -> float:
    """
    Compare the actual location words.

    Returns 0..1.
    """

    loc1 = clean_text(row1.get("location_text", ""))
    loc2 = clean_text(row2.get("location_text", ""))

    if not loc1 or not loc2:
        return 0.0

    return fuzz.token_set_ratio(
        loc1,
        loc2
    ) / 100.0


# ---------------------------------------------------------
# SUBCATEGORY COMPARISON
# ---------------------------------------------------------

def subcategory_similarity(
    row1: pd.Series,
    row2: pd.Series
) -> float:

    a = clean_text(row1.get("subcategory", ""))
    b = clean_text(row2.get("subcategory", ""))

    if not a or not b:
        return 0.0

    return fuzz.token_set_ratio(a, b) / 100.0


# ---------------------------------------------------------
# INFRASTRUCTURE COMPARISON
# ---------------------------------------------------------

def infrastructure_similarity(
    row1: pd.Series,
    row2: pd.Series
) -> float:

    a = clean_text(row1.get("infrastructure", ""))
    b = clean_text(row2.get("infrastructure", ""))

    if not a or not b:
        return 0.0

    return fuzz.token_set_ratio(a, b) / 100.0


# ---------------------------------------------------------
# GEOGRAPHIC DISTANCE
# ---------------------------------------------------------

def haversine_m(
    lat1,
    lon1,
    lat2,
    lon2
):

    values = [lat1, lon1, lat2, lon2]

    if any(pd.isna(x) for x in values):
        return None

    try:
        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)
    except Exception:
        return None

    r = 6371000

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(delta_lambda / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return r * c


# ---------------------------------------------------------
# PAIR SCORE
# ---------------------------------------------------------

def calculate_pair_score(
    row1: pd.Series,
    row2: pd.Series,
    semantic_score: float
) -> float:

    # Different complaint categories cannot represent
    # the same underlying issue.
    if str(row1.get("category")) != str(row2.get("category")):
        return 0.0

    location_score = location_similarity(
        row1,
        row2
    )

    sub_score = subcategory_similarity(
        row1,
        row2
    )

    infrastructure_score = infrastructure_similarity(
        row1,
        row2
    )

    distance = haversine_m(
        row1.get("lat"),
        row1.get("lon"),
        row2.get("lat"),
        row2.get("lon"),
    )

    # Geographic distance is only a SMALL bonus.
    geo_bonus = 0.0

    if distance is not None:

        if distance <= 100:
            geo_bonus = 0.05

        elif distance <= 250:
            geo_bonus = 0.03

        elif distance <= 500:
            geo_bonus = 0.01

    score = (
        semantic_score * 0.65
        + sub_score * 0.15
        + infrastructure_score * 0.10
        + location_score * 0.10
        + geo_bonus
    )

    return min(float(score), 1.0)


# ---------------------------------------------------------
# STRONG DUPLICATE DECISION
# ---------------------------------------------------------

def is_duplicate_pair(
    row1: pd.Series,
    row2: pd.Series,
    semantic_score: float,
    threshold: float
) -> tuple[bool, float]:

    if str(row1.get("category")) != str(row2.get("category")):
        return False, 0.0

    score = calculate_pair_score(
        row1,
        row2,
        semantic_score
    )

    loc = location_similarity(
        row1,
        row2
    )

    sub = subcategory_similarity(
        row1,
        row2
    )

    # Extremely strong semantic match can pass even
    # when location wording is slightly different.
    if semantic_score >= 0.92:
        return True, score

    # Otherwise we expect some location agreement.
    if (
        score >= threshold
        and loc >= 0.55
    ):
        return True, score

    # Another useful route:
    # same specific type of issue + strong location match.
    if (
        semantic_score >= 0.72
        and sub >= 0.75
        and loc >= 0.75
    ):
        return True, score

    return False, score


# ---------------------------------------------------------
# CREATE PAIRWISE MATRIX
# ---------------------------------------------------------

def create_pair_matrix(
    df: pd.DataFrame,
    threshold: float
):

    semantic_texts = [
        make_semantic_text(row)
        for _, row in df.iterrows()
    ]

    print("Creating multilingual embeddings...")

    embeddings = model.encode(
        semantic_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Because embeddings are normalized:
    # dot product = cosine similarity
    semantic_matrix = embeddings @ embeddings.T

    n = len(df)

    pair_matrix = np.zeros(
        (n, n),
        dtype=float
    )

    duplicate_matrix = np.zeros(
        (n, n),
        dtype=bool
    )

    for i in range(n):

        pair_matrix[i, i] = 1.0
        duplicate_matrix[i, i] = True

        for j in range(i + 1, n):

            duplicate, score = is_duplicate_pair(
                df.iloc[i],
                df.iloc[j],
                float(semantic_matrix[i, j]),
                threshold
            )

            pair_matrix[i, j] = score
            pair_matrix[j, i] = score

            duplicate_matrix[i, j] = duplicate
            duplicate_matrix[j, i] = duplicate

    return pair_matrix, duplicate_matrix


# ---------------------------------------------------------
# SAFER CLUSTERING
# ---------------------------------------------------------

def safe_clusters(
    df: pd.DataFrame,
    duplicate_matrix: np.ndarray
):

    """
    Avoid the chain problem:

    A ~ B
    B ~ C

    should NOT automatically mean A ~ C.

    A complaint may join a cluster only when it matches
    EVERY existing member of that cluster.
    """

    clusters: list[list[int]] = []

    for i in range(len(df)):

        placed = False

        for cluster in clusters:

            # Categories must match
            category = str(df.iloc[i].get("category"))

            cluster_category = str(
                df.iloc[cluster[0]].get("category")
            )

            if category != cluster_category:
                continue

            # COMPLETE-LINK style rule:
            # must be duplicate with every cluster member
            if all(
                duplicate_matrix[i, member]
                for member in cluster
            ):

                cluster.append(i)

                placed = True

                break

        if not placed:
            clusters.append([i])

    return clusters


# ---------------------------------------------------------
# ASSIGN CLUSTERS
# ---------------------------------------------------------

def assign_clusters(
    df: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD
) -> pd.DataFrame:

    print()
    print("Building duplicate similarity matrix...")

    _, duplicate_matrix = create_pair_matrix(
        df,
        threshold
    )

    print("Building safe clusters...")

    clusters = safe_clusters(
        df,
        duplicate_matrix
    )

    cluster_ids = [None] * len(df)

    for number, members in enumerate(
        clusters,
        start=1
    ):

        cid = f"cluster_{number:03d}"

        for index in members:
            cluster_ids[index] = cid

    result = df.copy()

    result["cluster_id"] = cluster_ids

    sizes = result[
        "cluster_id"
    ].value_counts()

    result["cluster_size"] = result[
        "cluster_id"
    ].map(sizes)

    return result


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def report(df: pd.DataFrame):

    if "true_cluster_id" not in df.columns:

        print(
            "true_cluster_id not available; "
            "cannot calculate metrics."
        )

        return {}

    tp = 0
    fp = 0
    fn = 0

    for i in range(len(df)):

        for j in range(i + 1, len(df)):

            predicted_same = (
                str(df.iloc[i]["cluster_id"])
                ==
                str(df.iloc[j]["cluster_id"])
            )

            actual_same = (
                str(df.iloc[i]["true_cluster_id"])
                ==
                str(df.iloc[j]["true_cluster_id"])
            )

            if predicted_same and actual_same:
                tp += 1

            elif predicted_same and not actual_same:
                fp += 1

            elif not predicted_same and actual_same:
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
    print("==============================")
    print("DUPLICATE DETECTION METRICS")
    print("==============================")

    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------
# QUICK THRESHOLD TUNING
# ---------------------------------------------------------

def tune(df: pd.DataFrame):

    thresholds = [
        0.68,
        0.72,
        0.75,
        0.78,
        0.80,
        0.82,
        0.85,
    ]

    best = None

    for threshold in thresholds:

        print()
        print(
            f"Testing threshold {threshold}"
        )

        clustered = assign_clusters(
            df,
            threshold=threshold
        )

        metrics = report(clustered)

        result = {
            "threshold": threshold,
            **metrics,
        }

        if (
            best is None
            or result["f1"] > best["f1"]
        ):
            best = result

    print()
    print("==============================")
    print("BEST THRESHOLD")
    print("==============================")

    for key, value in best.items():
        print(f"{key}: {value}")

    return best


# ---------------------------------------------------------
# LIVE COMPLAINT MATCHING
# ---------------------------------------------------------

def match_existing(
    new_row: dict,
    df: pd.DataFrame
) -> dict:

    """
    Compare one new complaint with existing complaints.
    """

    if len(df) == 0:

        return {
            "cluster_id": None,
            "is_duplicate": False,
            "best_match_id": None,
            "score": 0.0,
        }

    existing_texts = [
        make_semantic_text(row)
        for _, row in df.iterrows()
    ]

    new_series = pd.Series(new_row)

    new_text = make_semantic_text(
        new_series
    )

    embeddings = model.encode(
        existing_texts + [new_text],
        normalize_embeddings=True,
        show_progress_bar=False
    )

    existing_embeddings = embeddings[:-1]
    new_embedding = embeddings[-1]

    semantic_scores = (
        existing_embeddings
        @ new_embedding
    )

    best_score = 0.0
    best_index = None

    for i in range(len(df)):

        row = df.iloc[i]

        duplicate, score = is_duplicate_pair(
            new_series,
            row,
            float(semantic_scores[i]),
            DEFAULT_THRESHOLD
        )

        if duplicate and score > best_score:

            best_score = score
            best_index = i

    if best_index is None:

        return {
            "cluster_id": None,
            "is_duplicate": False,
            "best_match_id": None,
            "score": round(best_score, 4),
        }

    matched = df.iloc[best_index]

    return {
        "cluster_id": matched.get(
            "cluster_id"
        ),
        "is_duplicate": True,
        "best_match_id": matched.get("id"),
        "score": round(best_score, 4),
    }


# ---------------------------------------------------------
# RUN FILE
# ---------------------------------------------------------

if __name__ == "__main__":

    print("Reading extracted complaints...")

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Found {len(df)} complaints."
    )

    clustered = assign_clusters(
        df,
        DEFAULT_THRESHOLD
    )

    clustered.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        f"Saved clustered data to:"
        f" {OUTPUT_FILE}"
    )

    print(
        "Total clusters:",
        clustered[
            "cluster_id"
        ].nunique()
    )

    report(clustered)