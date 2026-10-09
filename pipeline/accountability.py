from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import TypedDict

import pandas as pd


class SLAResult(TypedDict):
    sla_hours: int
    sla_reason: str


def get_sla_hours(priority_score: float, category: str) -> SLAResult:
    category = str(category).strip().lower()

    if priority_score >= 85:
        return {"sla_hours": 6, "sla_reason": "Critical priority complaint."}

    if priority_score >= 70:
        return {"sla_hours": 12, "sla_reason": "High priority complaint."}

    if category in {"water_leak", "drainage", "road_damage"}:
        return {
            "sla_hours": 24,
            "sla_reason": "Infrastructure complaint requiring timely intervention.",
        }

    if priority_score >= 40:
        return {"sla_hours": 36, "sla_reason": "Medium priority complaint."}

    return {"sla_hours": 48, "sla_reason": "Normal priority complaint."}


def evaluate_issue(
    assigned_at: str | datetime,
    status: str,
    priority_score: float,
    category: str,
    now: datetime | None = None,
) -> dict[str, object]:
    if now is None:
        now = datetime.now()

    assigned_time = pd.to_datetime(assigned_at, errors="coerce")
    if pd.isna(assigned_time):
        raise ValueError("Invalid assigned_at timestamp.")

    assigned_time = assigned_time.to_pydatetime()
    status_clean = str(status).strip().lower()

    sla = get_sla_hours(priority_score, category)
    sla_hours = sla["sla_hours"]
    deadline = assigned_time + timedelta(hours=sla_hours)
    elapsed_hours = (now - assigned_time).total_seconds() / 3600

    closed_statuses = {"resolved", "closed", "completed"}
    is_closed = status_clean in closed_statuses
    overdue = not is_closed and now > deadline

    hours_overdue = 0.0
    if overdue:
        hours_overdue = (now - deadline).total_seconds() / 3600

    escalation_needed = overdue or (priority_score >= 90 and not is_closed)

    if is_closed:
        accountability_status = "RESOLVED"
    elif overdue:
        accountability_status = "OVERDUE"
    elif priority_score >= 85:
        accountability_status = "URGENT"
    else:
        accountability_status = "WITHIN_SLA"

    return {
        "sla_hours": sla_hours,
        "sla_reason": sla["sla_reason"],
        "deadline": deadline,
        "elapsed_hours": round(elapsed_hours, 2),
        "overdue": overdue,
        "hours_overdue": round(hours_overdue, 2),
        "escalation_needed": escalation_needed,
        "accountability_status": accountability_status,
    }


def apply_accountability(
    df: pd.DataFrame,
    now: datetime | None = None,
) -> pd.DataFrame:
    if now is None:
        now = datetime.now()

    required_columns = {"priority_score", "category", "status", "assigned_at"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df.copy()
    results: list[dict[str, object]] = []

    for _, row in data.iterrows():
        results.append(
            evaluate_issue(
                assigned_at=row["assigned_at"],
                status=row["status"],
                priority_score=float(row["priority_score"]),
                category=str(row["category"]),
                now=now,
            )
        )

    return pd.concat(
        [data.reset_index(drop=True), pd.DataFrame(results)],
        axis=1,
    )


def ward_performance(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"ward", "status", "overdue", "escalation_needed"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df.copy()
    data["status_clean"] = data["status"].astype(str).str.lower().str.strip()
    data["resolved"] = data["status_clean"].isin({"resolved", "closed", "completed"})

    performance = (
        data.groupby("ward")
        .agg(
            total_issues=("ward", "size"),
            resolved_issues=("resolved", "sum"),
            overdue_issues=("overdue", "sum"),
            escalation_issues=("escalation_needed", "sum"),
        )
        .reset_index()
    )

    performance["resolution_rate"] = (
        performance["resolved_issues"] / performance["total_issues"] * 100
    ).round(2)

    performance["overdue_rate"] = (
        performance["overdue_issues"] / performance["total_issues"] * 100
    ).round(2)

    performance["performance_score"] = (
        100
        - performance["overdue_rate"]
        - (performance["escalation_issues"] / performance["total_issues"] * 20)
    ).round(2).clip(lower=0, upper=100)

    return performance.sort_values(
        by=["performance_score", "resolution_rate"],
        ascending=[False, False],
    ).reset_index(drop=True)


def department_performance(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"department", "status", "overdue", "escalation_needed"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df.copy()
    data["resolved"] = (
        data["status"]
        .astype(str)
        .str.lower()
        .str.strip()
        .isin({"resolved", "closed", "completed"})
    )

    performance = (
        data.groupby("department")
        .agg(
            total_issues=("department", "size"),
            resolved_issues=("resolved", "sum"),
            overdue_issues=("overdue", "sum"),
            escalation_issues=("escalation_needed", "sum"),
        )
        .reset_index()
    )

    performance["resolution_rate"] = (
        performance["resolved_issues"] / performance["total_issues"] * 100
    ).round(2)

    performance["overdue_rate"] = (
        performance["overdue_issues"] / performance["total_issues"] * 100
    ).round(2)

    return performance.sort_values(
        by=["overdue_rate", "resolution_rate"],
        ascending=[True, False],
    ).reset_index(drop=True)


def main() -> None:
    now = datetime.now()

    demo_data = pd.DataFrame(
        [
            {
                "id": "CIV-101",
                "ward": "Ward 63",
                "category": "drainage",
                "priority_score": 91,
                "department": "Jaipur Municipal Corporation",
                "status": "Open",
                "assigned_at": now - timedelta(hours=20),
            },
            {
                "id": "CIV-102",
                "ward": "Ward 63",
                "category": "waste",
                "priority_score": 60,
                "department": "Jaipur Municipal Corporation",
                "status": "Resolved",
                "assigned_at": now - timedelta(hours=15),
            },
            {
                "id": "CIV-103",
                "ward": "Ward 21",
                "category": "road_damage",
                "priority_score": 76,
                "department": "PWD Rajasthan",
                "status": "Open",
                "assigned_at": now - timedelta(hours=8),
            },
        ]
    )

    accountability = apply_accountability(demo_data, now=now)

    print("=" * 70)
    print("CIVICPULSE - ACCOUNTABILITY DEMO")
    print("=" * 70)

    print(
        accountability[
            [
                "id", "ward", "priority_score", "status",
                "sla_hours", "elapsed_hours", "overdue",
                "escalation_needed", "accountability_status",
            ]
        ].to_string(index=False)
    )

    print("\nWARD PERFORMANCE")
    print(ward_performance(accountability).to_string(index=False))

    print("\nDEPARTMENT PERFORMANCE")
    print(department_performance(accountability).to_string(index=False))

    project_root = Path(__file__).resolve().parent.parent
    scored_path = project_root / "data" / "scored.csv"

    if scored_path.exists():
        real_df = pd.read_csv(scored_path, encoding="utf-8")
        print(f"\nLoaded {len(real_df)} rows from scored.csv.")
        print("Real SLA analysis also needs status, assigned_at and ward.")


if __name__ == "__main__":
    main()
