from __future__ import annotations

import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "extracted.csv"
OUTPUT_FILE = ROOT / "data" / "clustered.csv"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

TEXT_WEIGHT = 0.5
DIST_WEIGHT = 0.3
TIME_WEIGHT = 0.2

THRESHOLD = 0.72


model = SentenceTransformer(MODEL_NAME)


def haversine_m(lat1, lon1, lat2, lon2):
    if any(pd.isna(x) for x in [lat1, lon1, lat2, lon2]):
        return None

    r = 6371000

    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))

    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return r * c


def hours_apart(t1, t2):
    try:
        a = pd.to_datetime(t1, utc=True)
        b = pd.to_datetime(t2, utc=True)

        return abs((a - b).total_seconds()) / 3600

    except Exception:
        return None


def pair_score(row1, row2, emb1, emb2):
    if row1["category"] != row2["category"]:
        return 0.0

    text_score = cosine_similarity(
        emb1.reshape(1, -1),
        emb2.reshape(1, -1)
    )[0][0]

    distance = haversine_m(
        row1.get("lat"),
        row1.get("lon"),
        row2.get("lat"),
        row2.get("lon"),
    )

    time_gap = hours_apart(
        row1.get("timestamp"),
        row2.get("timestamp"),
    )

    if distance is None:
        distance_score = 0
    else:
        distance_score = math.exp(-distance / 150)

    if time_gap is None:
        time_score = 0
    else:
        time_score = math.exp(-time_gap / 72)

    score = (
        TEXT_WEIGHT * text_score
        + DIST_WEIGHT * distance_score
        + TIME_WEIGHT * time_score
    )

    return float(score)


def assign_clusters(df: pd.DataFrame) -> pd.DataFrame:
    texts = df["raw_text"].fillna("").astype(str).tolist()

    print("Creating multilingual embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    parent = list(range(len(df)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]

        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)

        if ra != rb:
            parent[rb] = ra

    print("Comparing complaints...")

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if df.iloc[i]["category"] != df.iloc[j]["category"]:
                continue

            score = pair_score(
                df.iloc[i],
                df.iloc[j],
                embeddings[i],
                embeddings[j],
            )

            if score >= THRESHOLD:
                union(i, j)

    roots = [find(i) for i in range(len(df))]

    root_to_cluster = {}
    next_id = 1

    cluster_ids = []

    for root in roots:
        if root not in root_to_cluster:
            root_to_cluster[root] = f"cluster_{next_id:03d}"
            next_id += 1

        cluster_ids.append(root_to_cluster[root])

    result = df.copy()

    result["cluster_id"] = cluster_ids

    sizes = result["cluster_id"].value_counts()

    result["cluster_size"] = result["cluster_id"].map(sizes)

    return result


def pairwise_report(df):
    if "true_cluster_id" not in df.columns:
        print("No true_cluster_id column found.")
        return

    tp = 0
    fp = 0
    fn = 0

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            pred_same = (
                df.iloc[i]["cluster_id"]
                == df.iloc[j]["cluster_id"]
            )

            true_same = (
                df.iloc[i]["true_cluster_id"]
                == df.iloc[j]["true_cluster_id"]
            )

            if pred_same and true_same:
                tp += 1

            elif pred_same and not true_same:
                fp += 1

            elif not pred_same and true_same:
                fn += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    print()
    print("Duplicate Detection Metrics")
    print("---------------------------")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")


if __name__ == "__main__":
    df = pd.read_csv(DATA_FILE)

    clustered = assign_clusters(df)

    clustered.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(f"Saved clustered data to: {OUTPUT_FILE}")
    print(f"Total clusters: {clustered['cluster_id'].nunique()}")

    pairwise_report(clustered)