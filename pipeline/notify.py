from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import TypedDict

import pandas as pd


class Notification(TypedDict):
    issue_id: str
    notification_type: str
    recipient: str
    subject: str
    message: str
    status: str


DEPARTMENT_CONTACTS = {
    "Jaipur Municipal Corporation": "JMC Control Room",
    "PHED Rajasthan": "PHED Jaipur Division",
    "PWD Rajasthan": "PWD Jaipur Division",
    "Jaipur DISCOM (JVVNL)": "JVVNL Jaipur Control Room",
}


def reminder_message(
    issue_id: str,
    category: str,
    department: str,
    priority_score: float,
    status: str,
    locality: str,
    deadline: str | datetime,
) -> Notification:
    recipient = DEPARTMENT_CONTACTS.get(department, department)
    deadline_text = pd.to_datetime(deadline).strftime("%d %b %Y, %I:%M %p")

    subject = f"Reminder: Civic Issue {issue_id} Requires Attention"

    message = (
        f"Issue ID: {issue_id}\n"
        f"Category: {category}\n"
        f"Locality: {locality}\n"
        f"Department: {department}\n"
        f"Priority: {priority_score:.0f}/100\n"
        f"Current Status: {status}\n"
        f"SLA Deadline: {deadline_text}\n\n"
        "This complaint is still unresolved and is approaching its response deadline. "
        "Please review and update the issue status."
    )

    return {
        "issue_id": issue_id,
        "notification_type": "REMINDER",
        "recipient": recipient,
        "subject": subject,
        "message": message,
        "status": "SIMULATED_SENT",
    }


def escalation_message(
    issue_id: str,
    category: str,
    department: str,
    priority_score: float,
    status: str,
    locality: str,
    hours_overdue: float,
) -> Notification:
    recipient = DEPARTMENT_CONTACTS.get(department, department)
    subject = f"ESCALATION: High Priority Civic Issue {issue_id}"

    message = (
        f"Issue ID: {issue_id}\n"
        f"Category: {category}\n"
        f"Locality: {locality}\n"
        f"Department: {department}\n"
        f"Priority: {priority_score:.0f}/100\n"
        f"Current Status: {status}\n"
        f"Hours Overdue: {hours_overdue:.1f}\n\n"
        "This complaint has exceeded its SLA or requires immediate attention because "
        "of its priority level.\nPlease review and take action immediately."
    )

    return {
        "issue_id": issue_id,
        "notification_type": "ESCALATION",
        "recipient": recipient,
        "subject": subject,
        "message": message,
        "status": "SIMULATED_SENT",
    }


def resolution_message(
    issue_id: str,
    category: str,
    department: str,
    locality: str,
) -> Notification:
    subject = f"Civic Issue {issue_id} Resolved"

    message = (
        f"Issue ID: {issue_id}\n"
        f"Category: {category}\n"
        f"Locality: {locality}\n"
        f"Department: {department}\n\n"
        "The civic complaint has been marked as resolved."
    )

    return {
        "issue_id": issue_id,
        "notification_type": "RESOLVED",
        "recipient": "Citizen / Dashboard",
        "subject": subject,
        "message": message,
        "status": "SIMULATED_SENT",
    }


def notify_issue(
    row: pd.Series,
    reminder_threshold_hours: float = 3.0,
) -> Notification | None:
    required_columns = {
        "id", "category", "department", "priority_score", "status",
        "deadline", "overdue", "hours_overdue", "escalation_needed",
    }

    missing = required_columns - set(row.index)
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    issue_id = str(row["id"])
    category = str(row["category"])
    department = str(row["department"])
    status = str(row["status"])
    priority_score = float(row["priority_score"])

    if "location_text" in row.index:
        locality = str(row["location_text"])
    elif "locality" in row.index:
        locality = str(row["locality"])
    elif "ward" in row.index:
        locality = str(row["ward"])
    else:
        locality = "Unknown locality"

    status_clean = status.lower().strip()

    if status_clean in {"resolved", "closed", "completed"}:
        return resolution_message(
            issue_id=issue_id,
            category=category,
            department=department,
            locality=locality,
        )

    if bool(row["escalation_needed"]):
        return escalation_message(
            issue_id=issue_id,
            category=category,
            department=department,
            priority_score=priority_score,
            status=status,
            locality=locality,
            hours_overdue=float(row["hours_overdue"]),
        )

    deadline = pd.to_datetime(row["deadline"], errors="coerce")
    if pd.isna(deadline):
        return None

    hours_remaining = (
        deadline.to_pydatetime() - datetime.now()
    ).total_seconds() / 3600

    if 0 <= hours_remaining <= reminder_threshold_hours:
        return reminder_message(
            issue_id=issue_id,
            category=category,
            department=department,
            priority_score=priority_score,
            status=status,
            locality=locality,
            deadline=deadline,
        )

    return None


def generate_notifications(
    df: pd.DataFrame,
    reminder_threshold_hours: float = 3.0,
) -> pd.DataFrame:
    output_columns = [
        "issue_id", "notification_type", "recipient",
        "subject", "message", "status",
    ]

    if df.empty:
        return pd.DataFrame(columns=output_columns)

    notifications: list[Notification] = []

    for _, row in df.iterrows():
        notification = notify_issue(row, reminder_threshold_hours)
        if notification is not None:
            notifications.append(notification)

    if not notifications:
        return pd.DataFrame(columns=output_columns)

    return pd.DataFrame(notifications)[output_columns]


def print_notification_log(notifications: pd.DataFrame) -> None:
    if notifications.empty:
        print("No notifications need to be sent.")
        return

    print("\n" + "=" * 70)
    print("NOTIFICATION LOG")
    print("=" * 70)

    for _, row in notifications.iterrows():
        print("\n" + "-" * 70)
        print(f"STATUS: {row['status']}")
        print(f"TYPE: {row['notification_type']}")
        print(f"ISSUE: {row['issue_id']}")
        print(f"TO: {row['recipient']}")
        print(f"SUBJECT: {row['subject']}")
        print()
        print(row["message"])


def main() -> None:
    now = datetime.now()

    demo_data = pd.DataFrame(
        [
            {
                "id": "CIV-101",
                "category": "drainage",
                "location_text": "Mansarovar",
                "department": "Jaipur Municipal Corporation",
                "priority_score": 91,
                "status": "Open",
                "deadline": now - timedelta(hours=8),
                "overdue": True,
                "hours_overdue": 8.0,
                "escalation_needed": True,
            },
            {
                "id": "CIV-102",
                "category": "road_damage",
                "location_text": "Malviya Nagar",
                "department": "PWD Rajasthan",
                "priority_score": 72,
                "status": "Open",
                "deadline": now + timedelta(hours=2),
                "overdue": False,
                "hours_overdue": 0.0,
                "escalation_needed": False,
            },
            {
                "id": "CIV-103",
                "category": "waste",
                "location_text": "C-Scheme",
                "department": "Jaipur Municipal Corporation",
                "priority_score": 45,
                "status": "Resolved",
                "deadline": now + timedelta(hours=20),
                "overdue": False,
                "hours_overdue": 0.0,
                "escalation_needed": False,
            },
        ]
    )

    notifications = generate_notifications(demo_data)

    print("=" * 70)
    print("CIVICPULSE - NOTIFICATION SYSTEM")
    print("=" * 70)

    print_notification_log(notifications)

    project_root = Path(__file__).resolve().parent.parent
    output_folder = project_root / "data"
    output_folder.mkdir(parents=True, exist_ok=True)

    log_path = output_folder / "notification_log.csv"
    notifications.to_csv(log_path, index=False, encoding="utf-8")

    print(f"\nNotification log saved to: {log_path}")


if __name__ == "__main__":
    main()
