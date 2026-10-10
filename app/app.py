from __future__ import annotations
from pipeline.analytics import route
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import folium
    from folium.plugins import HeatMap
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

from pipeline.geocode import geocode
from pipeline.geocode import geocode
process_one = None
LIVE_AI_AVAILABLE = False
from pipeline.score import explain, score, select_top
from pipeline.score import explain, score, select_top

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT_DIR / "data" / "complaints_raw.csv"
SCORED_PATH = ROOT_DIR / "data" / "scored.csv"
JAIPUR_CENTER = [26.9124, 75.7873]

st.set_page_config(
    page_title="CivicPulse | AI Civic Complaint Triage",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
html, body, [class*="css"] {font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}
.stApp {background:#000; color:#F5F5F7;}
.ios-card {background:rgba(28,28,30,.82); border:1px solid rgba(255,255,255,.10); border-radius:18px; padding:20px 22px; margin-bottom:14px;}
.ios-headline {font-size:28px; font-weight:750; color:#fff; letter-spacing:-.5px;}
.ios-subhead {font-size:14px; color:#8E8E93; margin-bottom:14px;}
.ios-label {font-size:11px; text-transform:uppercase; letter-spacing:.6px; font-weight:650; color:#8E8E93;}
.ios-val {font-size:20px; font-weight:700; color:#fff;}
.badge {display:inline-block; padding:4px 9px; border-radius:999px; font-size:12px; font-weight:650;}
.badge-red {background:rgba(255,69,58,.18); color:#FF453A;}
.badge-amber {background:rgba(255,159,10,.18); color:#FF9F0A;}
.badge-green {background:rgba(48,209,88,.18); color:#30D158;}
.badge-blue {background:rgba(10,132,255,.18); color:#0A84FF;}
.stButton>button {background:#0A84FF !important; color:white !important; border:0 !important; border-radius:12px !important; font-weight:650 !important;}
.stTabs [data-baseweb="tab-list"] {gap:6px; background:#1C1C1E; padding:4px; border-radius:14px;}
.stTabs [data-baseweb="tab"] {border-radius:10px; color:#9A9AA0; padding:0 14px;}
.stTabs [aria-selected="true"] {background:#3A3A3C !important; color:#fff !important;}
</style>
""",
    unsafe_allow_html=True,
)


def safe_text(value: object, default: str = "Unknown") -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        return default
@st.cache_data(show_spinner="Loading CivicPulse data...")
def load_historical_data():

    df = pd.read_csv(SCORED_PATH)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if "matched_name" not in df.columns:
        df["matched_name"] = df["location_text"]

    if "department" not in df.columns:
        df["department"] = df["category"].apply(
            lambda cat: route(cat, "")["department"]
        )

    if "explanation" not in df.columns:
        if "priority_explanation" in df.columns:
            df["explanation"] = df["priority_explanation"]
        else:
            df["explanation"] = ""

    return df


def build_live_row(text: str, historical: pd.DataFrame) -> pd.Series:

    # ------------------------------------------------
    # REAL LOCAL AI MODE
    # ------------------------------------------------
    if LIVE_AI_AVAILABLE and process_one is not None:

        try:
            result = process_one(text)

            extraction = result["extraction"]
            geocoding = result["geocoding"]
            priority = result["priority"]
            routing = result["routing"]
            duplicate = result["duplicate_detection"]

            return pd.Series({
                "id": result["complaint_id"],
                "raw_text": text,

                "category": extraction["category"],
                "subcategory": extraction["subcategory"],
                "location_text": extraction["location_text"],
                "severity": extraction["severity"],
                "duration_days": extraction["duration_days"],
                "infrastructure": extraction["infrastructure"],

                "lat": geocoding["lat"],
                "lon": geocoding["lon"],
                "matched_name": geocoding["matched_location"],
                "geocode_confidence": geocoding["confidence"],

                "priority_score": priority["score"],
                "explanation": priority["explanation"],

                "department": routing["department"],

                "cluster_id": result["cluster_id"],

                "is_duplicate": duplicate["is_duplicate"],
                "best_match_id": duplicate["best_match_id"],
                "duplicate_score": duplicate["score"],

                "ai_mode": "Local Ollama AI"
            })

        except Exception:
            pass


    # ------------------------------------------------
    # CLOUD FALLBACK MODE
    # ------------------------------------------------

    lower = text.lower()

    if any(word in lower for word in [
        "garbage", "kachra", "कचरा", "waste"
    ]):
        category = "waste"

    elif any(word in lower for word in [
        "drain", "nali", "naali", "sewer",
        "नाली", "waterlogging"
    ]):
        category = "drainage"

    elif any(word in lower for word in [
        "pothole", "road", "sadak", "सड़क"
    ]):
        category = "road_damage"

    elif any(word in lower for word in [
        "streetlight", "street light", "light"
    ]):
        category = "streetlight"

    elif any(word in lower for word in [
        "water leak", "pipe", "pipeline", "leakage"
    ]):
        category = "water_leak"

    else:
        category = "other"


    # Rough severity for public demo
    severity = 8 if any(word in lower for word in [
        "danger", "accident", "urgent",
        "overflow", "fire", "broken"
    ]) else 5


    # Try common Jaipur locations
    known_locations = [
        "Mansarovar",
        "Malviya Nagar",
        "Vaishali Nagar",
        "Jagatpura",
        "Raja Park",
        "Jhotwara",
        "Sanganer",
        "C-Scheme"
    ]

    location = None

    for place in known_locations:
        if place.lower() in lower:
            location = place
            break


    lat = None
    lon = None
    matched_name = location
    confidence = 0.0

    if location:
        try:
            lat, lon, matched_name, confidence = geocode(location)
        except Exception:
            pass


    department_info = route(category, "")


    # Simple explainable fallback priority
    priority_score = min(
        100,
        25 + severity * 6
    )


    return pd.Series({
        "id": "CLOUD-DEMO",
        "raw_text": text,

        "category": category,
        "subcategory": "",
        "location_text": location,
        "severity": severity,
        "duration_days": 0,
        "infrastructure": "",

        "lat": lat,
        "lon": lon,
        "matched_name": matched_name,
        "geocode_confidence": confidence,

        "priority_score": priority_score,

        "explanation":
            f"Cloud demo fallback: severity {severity}/10 "
            f"and category {category}.",

        "department":
            department_info["department"],

        "cluster_id": "cloud_demo",

        "is_duplicate": False,
        "best_match_id": None,
        "duplicate_score": 0.0,

        "ai_mode": "Cloud fallback"
    })
    result = process_one(text)
    extraction = result["extraction"]
    geocoding = result["geocoding"]
    priority = result["priority"]
    routing = result["routing"]
    duplicate = result["duplicate_detection"]

    row = pd.Series({
        "id": result["complaint_id"],
        "raw_text": text,
        "category": extraction["category"],
        "subcategory": extraction["subcategory"],
        "location_text": extraction["location_text"],
        "severity": extraction["severity"],
        "duration_days": extraction["duration_days"],
        "infrastructure": extraction["infrastructure"],

        "lat": geocoding["lat"],
        "lon": geocoding["lon"],
        "matched_name": geocoding["matched_location"],
        "geocode_confidence": geocoding["confidence"],

        "priority_score": priority["score"],
        "explanation": priority["explanation"],

        "department": routing["department"],

        "cluster_id": result["cluster_id"],

        "is_duplicate": duplicate["is_duplicate"],
        "best_match_id": duplicate["best_match_id"],
        "duplicate_score": duplicate["score"],
    })

    return row

def emerging_tables(df: pd.DataFrame):
    work = df.copy()
    work["date"] = pd.to_datetime(work["timestamp"], errors="coerce").dt.date
    category_counts = work["category"].value_counts().rename_axis("category").reset_index(name="complaints")
    daily_counts = work.groupby("date").size().reset_index(name="complaints")
    location_counts = (
        work.assign(location=work["matched_name"].fillna(work["true_location"]).fillna("Unknown"))
        .groupby("location").size().reset_index(name="complaints")
        .sort_values("complaints", ascending=False)
    )

    valid_ts = pd.to_datetime(work["timestamp"], errors="coerce")
    latest = valid_ts.max().normalize()
    recent_start = latest - pd.Timedelta(days=2)
    baseline_start = recent_start - pd.Timedelta(days=7)
    recent = work[valid_ts >= recent_start]
    baseline = work[(valid_ts >= baseline_start) & (valid_ts < recent_start)]

    def signal_frame(frame: pd.DataFrame, days: int, count_name: str) -> pd.DataFrame:
        loc = frame["matched_name"].fillna(frame["true_location"]).fillna("Unknown")
        temp = frame.assign(_location=loc)
        out = temp.groupby(["category", "_location"]).size().reset_index(name=count_name)
        out[count_name] = out[count_name] * (3.0 / days)
        return out

    recent_counts = signal_frame(recent, 3, "recent_3d")
    baseline_counts = signal_frame(baseline, 7, "baseline_3d_equiv")
    spikes = recent_counts.merge(baseline_counts, on=["category", "_location"], how="left").fillna({"baseline_3d_equiv": 0})
    spikes["spike_ratio"] = np.where(
        spikes["baseline_3d_equiv"] > 0,
        spikes["recent_3d"] / spikes["baseline_3d_equiv"],
        np.inf,
    )
    spikes = spikes.sort_values(["recent_3d", "spike_ratio"], ascending=[False, False])

    cluster_sizes = (
        work.groupby("cluster_id").size().reset_index(name="cluster_size")
        .sort_values("cluster_size", ascending=False)
    )
    return category_counts, daily_counts, location_counts, spikes, cluster_sizes


def waste_hotspots(df: pd.DataFrame) -> pd.DataFrame:
    waste = df[df["category"].eq("waste")].copy()
    if waste.empty:
        return pd.DataFrame(columns=["location", "complaints", "distinct_dates", "latest_complaint"])
    waste["location"] = waste["matched_name"].fillna(waste["true_location"]).fillna("Unknown")
    waste["complaint_date"] = pd.to_datetime(waste["timestamp"], errors="coerce").dt.date
    return (
        waste.groupby("location")
        .agg(
            complaints=("id", "count"),
            distinct_dates=("complaint_date", "nunique"),
            latest_complaint=("complaint_date", "max"),
            avg_priority=("priority_score", "mean"),
            lat=("lat", "mean"),
            lon=("lon", "mean"),
        )
        .reset_index()
        .sort_values(["complaints", "distinct_dates"], ascending=[False, False])
    )


try:
    scored_df = load_historical_data()
except Exception as exc:
    st.error(f"Could not load CivicPulse data: {exc}")
    st.stop()

st.markdown("<div class='ios-headline'>CivicPulse</div>", unsafe_allow_html=True)
st.markdown("<div class='ios-subhead'>Jaipur AI Civic Complaint Triage • real dataset • local-first geocoding • explainable priority scoring</div>", unsafe_allow_html=True)

tabs = st.tabs([
    "Live Triage",
    "Priority Queue",
    "Map View",
    "Emerging Issues",
    "Waste Hotspots",
    "Accountability",
    "Notifications",
    "System Metrics"
])

with tabs[0]:
    st.markdown("<div class='ios-subhead'>Paste an English, Hindi, or Hinglish complaint. CivicPulse will triage it using simple hackathon-safe rules.</div>", unsafe_allow_html=True)
    complaint = st.text_area(
        "Complaint text",
        placeholder="Example: Mansarovar sec 7 me naali overflow ho rahi hai since yesterday, road pe ganda pani bhar gaya.",
        height=120,
    )
    if st.button("Run Live Triage", type="primary"):
        if not complaint.strip():
            st.warning("Enter a complaint first.")
        else:
            row = build_live_row(complaint, scored_df)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Category", safe_text(row.get("category"), "other"))
            c2.metric("Location", safe_text(row.get("matched_name") or row.get("location_text")))
            c3.metric("Priority", f"{float(row.get('priority_score', 0)):.1f}/100")
            c4.metric("Department", safe_text(row.get("department")))
            st.markdown(
                f"""<div class='ios-card'>
                <div class='ios-label'>Triage explanation</div><div style='margin-top:6px'>{row.get('explanation')}</div><br>
<span class='badge badge-blue'>Severity {int(row.get('severity') or 0)}/10</span>                <span class='badge badge-amber'>Duration {float(row.get('duration_days') or 0):.1f} days</span>
<span class='badge badge-green'>Geocode confidence {float(row.get('geocode_confidence') or 0)*100:.0f}%</span>                </div>""",
                unsafe_allow_html=True,
            )
            if pd.notna(row.get("lat")) and pd.notna(row.get("lon")):
                st.caption(f"Coordinates: {float(row['lat']):.5f}, {float(row['lon']):.5f}")
            else:
                st.info("No Jaipur gazetteer location was detected in this complaint; it can be sent for manual location review.")

with tabs[1]:
    st.markdown("<div class='ios-subhead'>Top unique complaint clusters selected by the existing priority model.</div>", unsafe_allow_html=True)
    max_clusters = max(1, scored_df["cluster_id"].nunique())
    budget = st.slider("Daily response capacity", min_value=5, max_value=min(50, max_clusters), value=min(20, max_clusters), step=5)
    top_df = select_top(scored_df, budget)
    p1, p2, p3 = st.columns(3)
    p1.metric("Selected clusters", len(top_df))
    p2.metric("Average priority", f"{top_df['priority_score'].mean():.1f}" if not top_df.empty else "0")
    p3.metric("Highest priority", f"{top_df['priority_score'].max():.1f}" if not top_df.empty else "0")
    show_cols = ["id", "category", "matched_name", "cluster_id", "cluster_size", "priority_score", "department", "explanation"]
    st.dataframe(top_df[show_cols], use_container_width=True, hide_index=True)

with tabs[2]:
    st.markdown("<div class='ios-subhead'>Jaipur complaint map. Rows without valid coordinates are safely skipped.</div>", unsafe_allow_html=True)
    map_df = scored_df.dropna(subset=["lat", "lon"]).copy()
    if FOLIUM_AVAILABLE:
        m = folium.Map(location=JAIPUR_CENTER, zoom_start=12, tiles="OpenStreetMap")
        for _, row in map_df.iterrows():
            priority = float(row["priority_score"])
            color = "#FF453A" if priority >= 70 else "#FF9F0A" if priority >= 50 else "#30D158"
            popup = (
                f"<b>{safe_text(row['category'])}</b><br>"
                f"Location: {safe_text(row['matched_name'], safe_text(row['true_location']))}<br>"
                f"Priority: {priority:.1f}"
            )
            folium.CircleMarker(
                [float(row["lat"]), float(row["lon"])],
                radius=5, color=color, fill=True, fill_opacity=0.72, popup=popup,
            ).add_to(m)
        st_folium(m, width=None, height=520, use_container_width=True)
    else:
        st.info("Install folium and streamlit-folium for the interactive map.")
        st.map(map_df[["lat", "lon"]].rename(columns={"lat": "latitude", "lon": "longitude"}))

with tabs[3]:
    st.markdown("<div class='ios-subhead'>Category volume, daily trend, location concentration, duplicate clusters, and recent 3-day spike detection.</div>", unsafe_allow_html=True)
    category_counts, daily_counts, location_counts, spikes, cluster_sizes = emerging_tables(scored_df)

    mansarovar = spikes[(spikes["category"].eq("drainage")) & spikes["_location"].str.contains("Mansarovar|Madhyam Marg|Metro", case=False, na=False)]
    if not mansarovar.empty:
        recent_total = float(mansarovar["recent_3d"].sum())
        st.markdown(
            f"<div class='ios-card'><span class='badge badge-red'>Emerging issue detected</span> "
            f"<b>Mansarovar drainage spike</b> — {recent_total:.0f} normalized complaints in the recent 3-day window across Mansarovar-linked locations.</div>",
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Category counts")
        chart = alt.Chart(category_counts).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("complaints:Q", title="Complaints"),
            y=alt.Y("category:N", sort="-x", title=None),
        )
        st.altair_chart(chart, use_container_width=True)
    with c2:
        st.markdown("#### Daily complaint counts")
        chart = alt.Chart(daily_counts).mark_line(point=True).encode(
            x=alt.X("date:T", title="Date"), y=alt.Y("complaints:Q", title="Complaints")
        )
        st.altair_chart(chart, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Top locations")
        st.dataframe(location_counts.head(12), use_container_width=True, hide_index=True)
    with c4:
        st.markdown("#### Recent 3-day spikes vs previous 7-day baseline")
        spike_view = spikes.rename(columns={"_location": "location"}).copy()
        spike_view["spike_ratio"] = spike_view["spike_ratio"].replace(np.inf, np.nan).round(2)
        st.dataframe(spike_view.head(15), use_container_width=True, hide_index=True)

    st.markdown("#### Largest duplicate clusters")
    st.dataframe(cluster_sizes.head(15), use_container_width=True, hide_index=True)

with tabs[4]:
    st.markdown("<div class='ios-subhead'>Recurring solid-waste locations grouped by canonical Jaipur gazetteer match.</div>", unsafe_allow_html=True)
    hotspots = waste_hotspots(scored_df)
    recurring = hotspots[hotspots["distinct_dates"] >= 2]
    target = recurring[recurring["location"].isin(["Gaurav Tower", "SMS Hospital Gate 2"])]
    if not target.empty:
        names = ", ".join(target["location"].tolist())
        st.markdown(f"<div class='ios-card'><span class='badge badge-amber'>Recurring hotspots</span> Detected: <b>{names}</b></div>", unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        if FOLIUM_AVAILABLE and not hotspots.dropna(subset=["lat", "lon"]).empty:
            wh = hotspots.dropna(subset=["lat", "lon"])
            hm = folium.Map(location=JAIPUR_CENTER, zoom_start=12, tiles="CartoDB dark_matter")
            HeatMap([[r.lat, r.lon, r.complaints] for r in wh.itertuples()], radius=24, blur=18).add_to(hm)
            st_folium(hm, width=None, height=430, use_container_width=True)
        else:
            st.info("No mapped waste rows available.")
    with right:
        st.dataframe(
            hotspots[["location", "complaints", "distinct_dates", "latest_complaint", "avg_priority"]].head(15),
            use_container_width=True,
            hide_index=True,
        )
with tabs[5]:
    st.markdown(
        "<div class='ios-subhead'>Ward-level accountability, SLA tracking and escalation monitoring.</div>",
        unsafe_allow_html=True
    )

    accountability_path = ROOT_DIR / "data" / "accountability.csv"

    if accountability_path.exists():

        acc = pd.read_csv(accountability_path)

        total_issues = len(acc)
        overdue_count = int(acc["overdue"].sum())
        escalation_count = int(acc["escalation_needed"].sum())

        resolved_count = int(
            acc["status"]
            .astype(str)
            .str.lower()
            .isin(["resolved", "closed", "completed"])
            .sum()
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Issues", total_issues)
        c2.metric("Resolved", resolved_count)
        c3.metric("Overdue", overdue_count)
        c4.metric("Escalations", escalation_count)
        st.markdown("#### Active accountability queue")

        show_cols = [
            "id",
            "ward",
            "category",
            "department",
            "priority_score",
            "status",
            "sla_hours",
            "overdue",
            "escalation_needed",
            "accountability_status"
        ]

        available_cols = [
            col for col in show_cols
            if col in acc.columns
        ]

        queue = acc.sort_values(
            "priority_score",
            ascending=False
        )

        st.dataframe(
            queue[available_cols].head(30),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### Ward performance")

        ward_path = ROOT_DIR / "data" / "ward_performance.csv"

        if ward_path.exists():
            ward_df = pd.read_csv(ward_path)

            st.dataframe(
                ward_df,
                use_container_width=True,
                hide_index=True
            )

        st.markdown("#### Department performance")

        department_path = ROOT_DIR / "data" / "department_performance.csv"

        if department_path.exists():
            department_df = pd.read_csv(department_path)

            st.dataframe(
                department_df,
                use_container_width=True,
                hide_index=True
            )
    else:
        st.warning("accountability.csv not found.")
with tabs[6]:
    st.markdown("## Notifications")

    notification_path = ROOT_DIR / "data" / "notification_log.csv"

    if notification_path.exists():
        notifications = pd.read_csv(notification_path)

        st.success(f"Loaded {len(notifications)} notifications")

        if notifications.empty:
            st.warning("Notification file is empty.")

        else:
            total_notifications = len(notifications)

            escalation_count = int(
                (notifications["notification_type"] == "ESCALATION").sum()
            )

            reminder_count = int(
                (notifications["notification_type"] == "REMINDER").sum()
            )

            resolved_count = int(
                (notifications["notification_type"] == "RESOLVED").sum()
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Total Notifications", total_notifications)
            c2.metric("Escalations", escalation_count)
            c3.metric("Reminders", reminder_count)
            c4.metric("Resolved Updates", resolved_count)

            st.markdown("### Notification Log")

            show_cols = [
                "issue_id",
                "notification_type",
                "recipient",
                "subject",
                "status"
            ]

            available_cols = [
                col for col in show_cols
                if col in notifications.columns
            ]

            st.dataframe(
                notifications[available_cols].head(30),
                use_container_width=True,
                hide_index=True
            )

    else:
        st.error("notification_log.csv was not found.")
with tabs[7]:
    total = len(scored_df)
    unique_clusters = scored_df["cluster_id"].replace("", np.nan).nunique()
    duplicate_count = int(total - unique_clusters)
    avg_priority = float(scored_df["priority_score"].mean())
    high_priority = int((scored_df["priority_score"] >= 70).sum())
    geocode_success = float(scored_df["lat"].notna().mean() * 100)

    cols = st.columns(6)
    metrics = [
        ("Total complaints", f"{total}"),
        ("Unique clusters", f"{unique_clusters}"),
        ("Duplicate complaints", f"{duplicate_count}"),
        ("Avg priority", f"{avg_priority:.1f}"),
        ("High priority ≥70", f"{high_priority}"),
        ("Geocoding success", f"{geocode_success:.1f}%"),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)

    st.markdown("#### Category distribution")
    dist = scored_df["category"].value_counts().rename_axis("category").reset_index(name="complaints")
    chart = alt.Chart(dist).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("complaints:Q", title="Complaints"),
        y=alt.Y("category:N", sort="-x", title=None),
        tooltip=["category", "complaints"],
    )
    st.altair_chart(chart, use_container_width=True)
