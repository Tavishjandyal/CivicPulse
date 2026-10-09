import pandas as pd
from pathlib import Path

from analytics import route
from accountability import (
    apply_accountability,
    ward_performance,
    department_performance,
)
from notify import generate_notifications


ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "data" / "scored.csv"

ACCOUNTABILITY_OUTPUT = ROOT / "data" / "accountability.csv"
WARD_OUTPUT = ROOT / "data" / "ward_performance.csv"
DEPARTMENT_OUTPUT = ROOT / "data" / "department_performance.csv"
NOTIFICATION_OUTPUT = ROOT / "data" / "notification_log.csv"


# Read our real 200 complaints
df = pd.read_csv(INPUT)


# -------------------------------------------------
# ADD DEPARTMENT
# -------------------------------------------------

def get_department(row):

    result = route(
        row["category"],
        row.get("subcategory", "")
    )

    return result["department"]


df["department"] = df.apply(
    get_department,
    axis=1
)


# -------------------------------------------------
# SLA + OVERDUE + ESCALATION
# -------------------------------------------------

accountability = apply_accountability(df)


# -------------------------------------------------
# WARD PERFORMANCE
# -------------------------------------------------

wards = ward_performance(accountability)


# -------------------------------------------------
# DEPARTMENT PERFORMANCE
# -------------------------------------------------

departments = department_performance(
    accountability
)


# -------------------------------------------------
# NOTIFICATIONS
# -------------------------------------------------

notifications = generate_notifications(
    accountability
)


# -------------------------------------------------
# SAVE EVERYTHING
# -------------------------------------------------

accountability.to_csv(
    ACCOUNTABILITY_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

wards.to_csv(
    WARD_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

departments.to_csv(
    DEPARTMENT_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

notifications.to_csv(
    NOTIFICATION_OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


# -------------------------------------------------
# SHOW RESULTS
# -------------------------------------------------

print()
print("======================================")
print("REAL CIVICPULSE ACCOUNTABILITY COMPLETE")
print("======================================")

print()
print("Total complaints:", len(accountability))

print(
    "Overdue:",
    int(accountability["overdue"].sum())
)

print(
    "Escalations:",
    int(
        accountability[
            "escalation_needed"
        ].sum()
    )
)

print()
print("WARD PERFORMANCE")
print(
    wards.to_string(index=False)
)

print()
print("DEPARTMENT PERFORMANCE")
print(
    departments.to_string(index=False)
)

print()
print(
    "Notifications generated:",
    len(notifications)
)

print()
print("Files created:")
print("data/accountability.csv")
print("data/ward_performance.csv")
print("data/department_performance.csv")
print("data/notification_log.csv")