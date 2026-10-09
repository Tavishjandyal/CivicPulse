from __future__ import annotations

from pathlib import Path
from typing import TypedDict
import math

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.cluster import DBSCAN


class RouteResult(TypedDict):
    department: str
    reason: str


def route(category: str, subcategory: str = "") -> RouteResult:
    category = str(category).strip().lower()
    subcategory = str(subcategory).strip().lower()
    text = f"{category} {subcategory}"

    electrical_keywords = [
        "electric", "electrical", "wire", "live wire", "fallen wire",
        "open wire", "transformer", "electric pole", "power line",
        "spark", "shock",
    ]

    if any(word in text for word in electrical_keywords):
        return {
            "department": "Jaipur DISCOM (JVVNL)",
            "reason": "Electrical hazards and electricity distribution infrastructure are handled by Jaipur DISCOM.",
        }

    if category == "road_damage":
        return {
            "department": "PWD Rajasthan",
            "reason": "Road damage, potholes and road maintenance are routed to Rajasthan PWD.",
        }

    if category == "water_leak":
        return {
            "department": "PHED Rajasthan",
            "reason": "Water supply pipelines and public water leakage are handled by PHED Rajasthan.",
        }

    if category == "waste":
        return {
            "department": "Jaipur Municipal Corporation",
            "reason": "Garbage collection, waste dumping and solid-waste management are municipal responsibilities.",
        }

    if category == "drainage":
        return {
            "department": "Jaipur Municipal Corporation",
            "reason": "Drain blockage, sewer overflow and local waterlogging are handled by the municipal corporation.",
        }

    if category == "streetlight":
        return {
            "department": "Jaipur Municipal Corporation",
            "reason": "Streetlight maintenance is handled by the municipal corporation.",
        }

    return {
        "department": "Jaipur Municipal Corporation",
        "reason": "No specialised department rule matched, so the complaint is routed to the municipal corporation for initial handling.",
    }


def trend_alerts(df: pd.DataFrame, window_days: int = 3) -> pd.DataFrame:
    output_columns = [
        "locality", "category", "recent_count",
        "baseline", "ratio", "p_value",
    ]

    if df.empty:
        return pd.DataFrame(columns=output_columns)

    if window_days < 1:
        raise ValueError("window_days must be at least 1.")

    data = df.copy()

    if "location_text" in data.columns:
        locality_column = "location_text"
    elif "locality" in data.columns:
        locality_column = "locality"
    elif "true_location" in data.columns:
        locality_column = "true_location"
    else:
        raise ValueError("No locality column found. Expected location_text.")

    required_columns = {locality_column, "category", "timestamp"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=[locality_column, "category", "timestamp"])

    if data.empty:
        return pd.DataFrame(columns=output_columns)

    data["date"] = data["timestamp"].dt.normalize()
    first_date = data["date"].min()
    last_date = data["date"].max()
    recent_start = last_date - pd.Timedelta(days=window_days - 1)

    daily_counts = (
        data.groupby([locality_column, "category", "date"])
        .size()
        .reset_index(name="count")
    )

    results: list[dict[str, object]] = []

    for (locality, category), group in daily_counts.groupby([locality_column, "category"]):
        full_dates = pd.date_range(start=first_date, end=last_date, freq="D")
        counts = (
            group.set_index("date")["count"]
            .reindex(full_dates, fill_value=0)
            .astype(float)
        )

        recent_counts = counts[counts.index >= recent_start]
        history = counts[counts.index < recent_start]

        if len(history) < window_days:
            continue

        recent_count = int(recent_counts.sum())

        historical_windows = (
            history.rolling(window=window_days).sum().dropna()
        )

        if historical_windows.empty:
            continue

        baseline = float(historical_windows.mean())
        baseline_std = float(historical_windows.std(ddof=0))

        if baseline_std > 0:
            z_score = (recent_count - baseline) / baseline_std
        elif recent_count > baseline:
            z_score = math.inf
        else:
            z_score = 0.0

        if baseline > 0:
            p_value = float(poisson.sf(recent_count - 1, baseline))
        else:
            p_value = 0.0 if recent_count > 0 else 1.0

        if baseline > 0:
            ratio = recent_count / baseline
        elif recent_count > 0:
            ratio = math.inf
        else:
            ratio = 0.0

        if z_score > 3 or p_value < 0.01:
            results.append(
                {
                    "locality": str(locality),
                    "category": str(category),
                    "recent_count": recent_count,
                    "baseline": round(baseline, 2),
                    "ratio": round(ratio, 2) if math.isfinite(ratio) else math.inf,
                    "p_value": round(p_value, 6),
                }
            )

    if not results:
        return pd.DataFrame(columns=output_columns)

    alerts = pd.DataFrame(results).sort_values(
        by=["p_value", "ratio", "recent_count"],
        ascending=[True, False, False],
    )

    return alerts[output_columns].reset_index(drop=True)


def waste_hotspots(df: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "cluster_id", "centroid_lat", "centroid_lon",
        "complaint_count", "distinct_days",
        "first_seen", "last_seen",
    ]

    if df.empty:
        return pd.DataFrame(columns=output_columns)

    required = {"category", "lat", "lon", "timestamp"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    waste = df[
        df["category"].astype(str).str.strip().str.lower().eq("waste")
    ].copy()

    if waste.empty:
        return pd.DataFrame(columns=output_columns)

    waste["lat"] = pd.to_numeric(waste["lat"], errors="coerce")
    waste["lon"] = pd.to_numeric(waste["lon"], errors="coerce")
    waste["timestamp"] = pd.to_datetime(waste["timestamp"], errors="coerce")

    waste = waste.dropna(subset=["lat", "lon", "timestamp"])
    waste = waste[
        waste["lat"].between(-90, 90)
        & waste["lon"].between(-180, 180)
    ].copy()

    if waste.empty:
        return pd.DataFrame(columns=output_columns)

    earth_radius_m = 6_371_000.0
    eps_radians = 150.0 / earth_radius_m
    coordinates = np.radians(waste[["lat", "lon"]].to_numpy())

    model = DBSCAN(
        eps=eps_radians,
        min_samples=2,
        metric="haversine",
        algorithm="ball_tree",
    )

    waste["hotspot_cluster"] = model.fit_predict(coordinates)
    waste = waste[waste["hotspot_cluster"] != -1].copy()

    if waste.empty:
        return pd.DataFrame(columns=output_columns)

    waste["date"] = waste["timestamp"].dt.date

    hotspots = (
        waste.groupby("hotspot_cluster")
        .agg(
            centroid_lat=("lat", "mean"),
            centroid_lon=("lon", "mean"),
            complaint_count=("hotspot_cluster", "size"),
            distinct_days=("date", "nunique"),
            first_seen=("timestamp", "min"),
            last_seen=("timestamp", "max"),
        )
        .reset_index()
        .rename(columns={"hotspot_cluster": "cluster_id"})
    )

    hotspots["centroid_lat"] = hotspots["centroid_lat"].round(6)
    hotspots["centroid_lon"] = hotspots["centroid_lon"].round(6)

    hotspots = hotspots.sort_values(
        by=["distinct_days", "complaint_count"],
        ascending=[False, False],
    )

    return hotspots[output_columns].reset_index(drop=True)


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data" / "clustered.csv"

    print("=" * 70)
    print("CIVICPULSE - P3 ANALYTICS")
    print("=" * 70)

    test_cases = [
        ("road_damage", "large pothole"),
        ("water_leak", "pipe burst"),
        ("waste", "garbage dumping"),
        ("drainage", "blocked drain"),
        ("streetlight", "light not working"),
        ("other", "fallen live electric wire"),
    ]

    for category, subcategory in test_cases:
        result = route(category, subcategory)
        print(f"\n{category} / {subcategory}")
        print(f"Department: {result['department']}")
        print(f"Reason: {result['reason']}")

    if not csv_path.exists():
        print("\nclustered.csv not found. route() demo completed.")
        return

    df = pd.read_csv(csv_path, encoding="utf-8")

    print("\nTREND ALERTS")
    alerts = trend_alerts(df, window_days=3)
    print("No unusual trends found." if alerts.empty else alerts.to_string(index=False))

    print("\nWASTE HOTSPOTS")
    hotspots = waste_hotspots(df)
    print("No waste hotspots found." if hotspots.empty else hotspots.to_string(index=False))


if __name__ == "__main__":
    main()
