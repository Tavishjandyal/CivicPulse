import math
import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "data" / "extracted.csv"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

df = pd.read_csv(FILE)

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("Creating embeddings...")
embeddings = model.encode(
    df["raw_text"].fillna("").astype(str).tolist(),
    normalize_embeddings=True,
    show_progress_bar=True,
)

# Because embeddings are normalized, dot product = cosine similarity
text_sim = embeddings @ embeddings.T


def haversine_m(lat1, lon1, lat2, lon2):
    if any(pd.isna(x) for x in [lat1, lon1, lat2, lon2]):
        return None

    r = 6371000

    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))

    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )

    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def evaluate(text_weight, dist_weight, time_weight, threshold):

    parent = list(range(len(df)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra = find(a)
        rb = find(b)

        if ra != rb:
            parent[rb] = ra

    for i in range(len(df)):
        for j in range(i + 1, len(df)):

            # Different categories cannot be duplicates
            if df.iloc[i]["category"] != df.iloc[j]["category"]:
                continue

            ts = float(text_sim[i][j])

            distance = haversine_m(
                df.iloc[i].get("lat"),
                df.iloc[i].get("lon"),
                df.iloc[j].get("lat"),
                df.iloc[j].get("lon"),
            )

            if distance is None:
                ds = 0
            else:
                ds = math.exp(-distance / 150)

            try:
                t1 = pd.to_datetime(df.iloc[i]["timestamp"], utc=True)
                t2 = pd.to_datetime(df.iloc[j]["timestamp"], utc=True)

                hours = abs((t1 - t2).total_seconds()) / 3600
                times = math.exp(-hours / 72)

            except Exception:
                times = 0

            score = (
                text_weight * ts
                + dist_weight * ds
                + time_weight * times
            )

            if score >= threshold:
                union(i, j)

    predicted = [find(i) for i in range(len(df))]

    tp = fp = fn = 0

    for i in range(len(df)):
        for j in range(i + 1, len(df)):

            predicted_same = predicted[i] == predicted[j]

            actual_same = (
                str(df.iloc[i]["true_cluster_id"])
                == str(df.iloc[j]["true_cluster_id"])
            )

            if predicted_same and actual_same:
                tp += 1

            elif predicted_same and not actual_same:
                fp += 1

            elif not predicted_same and actual_same:
                fn += 1

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )

    return precision, recall, f1


weight_sets = [
    (0.60, 0.25, 0.15),
    (0.70, 0.20, 0.10),
    (0.75, 0.15, 0.10),
    (0.80, 0.10, 0.10),
    (0.85, 0.10, 0.05),
    (0.90, 0.05, 0.05),
]

thresholds = [
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]

best = None

print()
print("Testing combinations...")
print()

for weights in weight_sets:

    for threshold in thresholds:

        p, r, f1 = evaluate(
            weights[0],
            weights[1],
            weights[2],
            threshold,
        )

        print(
            f"text={weights[0]:.2f} "
            f"distance={weights[1]:.2f} "
            f"time={weights[2]:.2f} "
            f"threshold={threshold:.2f} "
            f"| P={p:.3f} R={r:.3f} F1={f1:.3f}"
        )

        if best is None or f1 > best["f1"]:
            best = {
                "text": weights[0],
                "distance": weights[1],
                "time": weights[2],
                "threshold": threshold,
                "precision": p,
                "recall": r,
                "f1": f1,
            }


print()
print("==============================")
print("BEST SETTINGS")
print("==============================")

for key, value in best.items():
    print(f"{key}: {value}")