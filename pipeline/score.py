from __future__ import annotations

import json
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "severity": 0.35,
    "duplicates": 0.25,
    "duration": 0.20,
    "recurrence": 0.10,
    "criticality": 0.10,
}

# Conservative defaults; unknown categories land near the middle rather than at 0.
CATEGORY_CRITICALITY = {
    "manhole": 1.00,
    "electrical_hazard": 0.95,
    "water_leak": 0.80,
    "drainage": 0.78,
    "waterlogging": 0.75,
    "pothole": 0.70,
    "traffic_signal": 0.65,
    "water_supply": 0.62,
    "stray_animals": 0.55,
    "waste": 0.50,
    "streetlight": 0.35,
    "park_maintenance": 0.25,
}


def _pick_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _minmax(series: pd.Series, *, floor_zero: bool = True) -> pd.Series:
    """Min-max normalize to [0,1], robust to all-null/constant inputs."""
    s = pd.to_numeric(series, errors="coerce")
    if floor_zero:
        s = s.clip(lower=0)
    if s.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)
    lo = float(s.min(skipna=True))
    hi = float(s.max(skipna=True))
    if np.isclose(lo, hi):
        # A constant positive factor should still count, but not dominate.
        val = 1.0 if hi > 0 else 0.0
        return pd.Series(val, index=series.index, dtype=float)
    return ((s - lo) / (hi - lo)).fillna(0.0).clip(0, 1)


def _validate_weights(weights: Optional[Mapping[str, float]]) -> Dict[str, float]:
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    missing = set(DEFAULT_WEIGHTS) - set(w)
    extra = set(w) - set(DEFAULT_WEIGHTS)
    if missing or extra:
        raise ValueError(
            f"weights must have exactly {sorted(DEFAULT_WEIGHTS)}; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    vals = {k: float(v) for k, v in w.items()}
    if any(v < 0 for v in vals.values()):
        raise ValueError("weights cannot be negative")
    total = sum(vals.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return {k: v / total for k, v in vals.items()}


def _factor_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    severity_col = _pick_col(df, "severity", "true_severity")
    duration_col = _pick_col(df, "duration_days", "duration", "true_duration_days")
    cluster_col = _pick_col(df, "cluster_id", "true_cluster_id")
    category_col = _pick_col(df, "category", "true_category")
    location_col = _pick_col(df, "location_text", "matched_name", "location", "true_location")
    timestamp_col = _pick_col(df, "timestamp", "created_at", "reported_at")
    severity_raw = (
        pd.to_numeric(df[severity_col], errors="coerce")
        if severity_col
        else pd.Series(0, index=df.index)
    )
    # Severity is usually on a 1-10 scale. Respect that scale when possible.
    if severity_raw.notna().any() and severity_raw.max(skipna=True) <= 10:
        out["severity"] = (severity_raw.fillna(0) / 10.0).clip(0, 1)
    else:
        out["severity"] = _minmax(severity_raw)

    if cluster_col:
        cluster_keys = df[cluster_col].astype(object).copy()
        missing = cluster_keys.isna() | cluster_keys.astype(str).str.strip().eq("")
        cluster_keys.loc[missing] = [f"__row_{i}" for i in df.index[missing]]
        dup_count = cluster_keys.map(cluster_keys.value_counts())
    else:
        dup_count = pd.Series(1, index=df.index)
    out["duplicates"] = _minmax(dup_count)

    duration_raw = (
        pd.to_numeric(df[duration_col], errors="coerce").fillna(0)
        if duration_col
        else pd.Series(0, index=df.index)
    )
    out["duration"] = _minmax(duration_raw)

    # Recurrence = count of complaints at same location inside the trailing 30-day
    # window relative to the latest timestamp in the dataframe. This is row-level,
    # not cluster-level, so repeated issues across different clusters still count.
    recurrence_count = pd.Series(0, index=df.index, dtype=float)
    if location_col and timestamp_col:
        ts = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)
        if ts.notna().any():
            cutoff = ts.max() - pd.Timedelta(days=30)
            loc = (
                df[location_col]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
            )
            valid = ts.ge(cutoff) & loc.ne("")
            counts = loc[valid].value_counts()
            recurrence_count = loc.map(counts).fillna(0).astype(float)
    elif location_col:
        loc = df[location_col].fillna("").astype(str).str.strip().str.lower()
        counts = loc[loc.ne("")].value_counts()
        recurrence_count = loc.map(counts).fillna(0).astype(float)
    out["recurrence"] = _minmax(recurrence_count)

    if category_col:
        cat = df[category_col].fillna("").astype(str).str.strip().str.lower()
        out["criticality"] = cat.map(CATEGORY_CRITICALITY).fillna(0.5).astype(float)
    else:
        out["criticality"] = 0.5

    # Keep useful raw values for explanation/debugging.
    out["_dup_count"] = dup_count.astype(float)
    out["_duration_days"] = duration_raw.astype(float)
    out["_recurrence_count"] = recurrence_count.astype(float)
    return out


def score(df: pd.DataFrame, weights: Optional[Mapping[str, float]] = None) -> pd.DataFrame:
    """Add priority_score (0-100) and JSON score_breakdown; keep all input columns."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    w = _validate_weights(weights)
    factors = _factor_frame(df)
    result = df.copy()

    contributions = pd.DataFrame(index=df.index)
    for factor, weight in w.items():
        contributions[factor] = factors[factor] * weight * 100.0

    result["priority_score"] = contributions.sum(axis=1).round(2).clip(0, 100)

    def make_breakdown(i):
        payload = {
            factor: {
                "normalized": round(float(factors.at[i, factor]), 4),
                "weight": round(float(w[factor]), 4),
                "contribution": round(float(contributions.at[i, factor]), 2),
            }
            for factor in DEFAULT_WEIGHTS
        }
        payload["raw"] = {
            "duplicate_count": int(factors.at[i, "_dup_count"]),
            "duration_days": round(float(factors.at[i, "_duration_days"]), 2),
            "recurrence_count_30d": int(factors.at[i, "_recurrence_count"]),
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    result["score_breakdown"] = [make_breakdown(i) for i in result.index]
    return result


def select_top(df: pd.DataFrame, budget_n: int) -> pd.DataFrame:
    """Return the best N issues, one representative row per cluster."""
    if budget_n < 0:
        raise ValueError("budget_n must be >= 0")
    if budget_n == 0:
        return df.head(0).copy()

    scored = df.copy() if "priority_score" in df.columns else score(df)
    cluster_col = _pick_col(scored, "true_cluster_id", "cluster_id")

    if cluster_col:
        cluster_key = scored[cluster_col].copy()
        missing = cluster_key.isna() | cluster_key.astype(str).str.strip().eq("")
        cluster_key = cluster_key.astype(object)
        cluster_key.loc[missing] = [f"__row_{i}" for i in scored.index[missing]]
        tmp = scored.assign(_cluster_key=cluster_key)
        # Stable tie-breaking: score desc, then timestamp desc if available.
        sort_cols = ["priority_score"]
        ascending = [False]
        ts_col = _pick_col(tmp, "timestamp", "created_at", "reported_at")
        if ts_col:
            tmp = tmp.assign(_sort_ts=pd.to_datetime(tmp[ts_col], errors="coerce", utc=True))
            sort_cols.append("_sort_ts")
            ascending.append(False)
        tmp = tmp.sort_values(sort_cols, ascending=ascending, kind="mergesort")
        top = tmp.drop_duplicates("_cluster_key", keep="first").head(budget_n)
        return top.drop(columns=[c for c in ["_cluster_key", "_sort_ts"] if c in top.columns])

    return scored.sort_values("priority_score", ascending=False, kind="mergesort").head(budget_n)


def sensitivity(
    df: pd.DataFrame,
    weight_sets: Sequence[Mapping[str, float]],
) -> pd.DataFrame:
    """Compare cluster rankings under multiple weight configurations.

    Returns one row per cluster per weight set with score, rank, baseline_rank,
    rank_change, and the normalized weights used.
    """
    if not weight_sets:
        return pd.DataFrame(
            columns=[
                "weight_set",
                "cluster_id",
                "priority_score",
                "rank",
                "baseline_rank",
                "rank_change",
                "weights",
            ]
        )

    cluster_col = _pick_col(df, "true_cluster_id", "cluster_id")
    id_col = _pick_col(df, "id", "complaint_id")
    all_rows = []
    baseline_ranks = None

    for idx, ws in enumerate(weight_sets):
        w = _validate_weights(ws)
        scored = score(df, w)

        if cluster_col:
            cluster_key = scored[cluster_col].fillna("").astype(str)
            cluster_key = cluster_key.where(cluster_key.str.strip().ne(""), scored.index.map(lambda i: f"__row_{i}"))
        elif id_col:
            cluster_key = scored[id_col].astype(str)
        else:
            cluster_key = scored.index.map(lambda i: f"row_{i}")

        tmp = scored.assign(_cluster_id=cluster_key)
        # Cluster score = best representative row score. Because duplicate/recurrence
        # factors are computed globally first, every row in a cluster is evaluated
        # in full context before this reduction.
        agg = (
            tmp.groupby("_cluster_id", as_index=False)["priority_score"]
            .max()
            .sort_values(["priority_score", "_cluster_id"], ascending=[False, True], kind="mergesort")
            .reset_index(drop=True)
        )
        agg["rank"] = np.arange(1, len(agg) + 1)
        ranks = dict(zip(agg["_cluster_id"], agg["rank"]))
        if baseline_ranks is None:
            baseline_ranks = ranks.copy()

        for _, r in agg.iterrows():
            base_rank = int(baseline_ranks[r["_cluster_id"]])
            rank = int(r["rank"])
            all_rows.append(
                {
                    "weight_set": idx,
                    "cluster_id": r["_cluster_id"],
                    "priority_score": round(float(r["priority_score"]), 2),
                    "rank": rank,
                    "baseline_rank": base_rank,
                    # Positive means the issue moved up (e.g. baseline 10 -> rank 4 = +6).
                    "rank_change": base_rank - rank,
                    "weights": json.dumps(w, separators=(",", ":")),
                }
            )

    return pd.DataFrame(all_rows)


def explain(row: pd.Series) -> str:
    """Return a one-line plain-English explanation for a scored row."""
    if "score_breakdown" not in row or pd.isna(row["score_breakdown"]):
        return f"Priority score {float(row.get('priority_score', 0)):.1f}/100 based on the configured factors."

    try:
        b = json.loads(row["score_breakdown"])
    except Exception:
        return f"Priority score {float(row.get('priority_score', 0)):.1f}/100 based on the configured factors."

    labels = {
        "severity": "severity",
        "duplicates": "multiple similar complaints",
        "duration": "long duration",
        "recurrence": "repeated reports at this location",
        "criticality": "category criticality",
    }
    ranked = sorted(
        ((k, float(b.get(k, {}).get("contribution", 0))) for k in DEFAULT_WEIGHTS),
        key=lambda x: x[1],
        reverse=True,
    )
    top = [labels[k] for k, v in ranked if v > 0][:3]
    score_val = float(row.get("priority_score", 0))
    if not top:
        return f"Priority score {score_val:.1f}/100 because none of the measured factors is elevated."
    if len(top) == 1:
        why = top[0]
    elif len(top) == 2:
        why = f"{top[0]} and {top[1]}"
    else:
        why = f"{top[0]}, {top[1]}, and {top[2]}"
    return f"Priority score {score_val:.1f}/100, driven mainly by {why}."


__all__ = [
    "DEFAULT_WEIGHTS",
    "CATEGORY_CRITICALITY",
    "score",
    "select_top",
    "sensitivity",
    "explain",
]
