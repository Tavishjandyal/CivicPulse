"""Jaipur-focused geocoding with fuzzy gazetteer matching and Nominatim fallback.

Public API:
    geocode(location_text) -> (lat, lon, matched_name, confidence)

The local gazetteer is always tried first. If its confidence is below 0.60,
Nominatim is queried at most once per second, and responses are cached on disk.
"""

from __future__ import annotations

import csv
import json
import re
import threading
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from rapidfuzz import fuzz


ROOT_DIR = Path(__file__).resolve().parents[1]
GAZETTEER_PATH = ROOT_DIR / "data" / "gazetteer.csv"
CACHE_PATH = ROOT_DIR / "data" / "nominatim_cache.json"

FUZZY_THRESHOLD = 0.60
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "jaipur-complaints-geocoder/1.0"

# Nominatim policy: never issue requests faster than one per second.
_rate_lock = threading.Lock()
_last_request_at = 0.0
_cache_lock = threading.Lock()


_NORMALIZATION_REPLACEMENTS = {
    "opp.": "opposite",
    "opp ": "opposite ",
    "nr ": "near ",
    "nr. ": "near ",
    "nagar": "nagar",
    "ngr": "nagar",
    "sec.": "sector",
    "sec ": "sector ",
    "rd.": "road",
    "rd ": "road ",
    "stn": "station",
    "hosp": "hospital",
}


def _normalize(text: str) -> str:
    """Normalize Latin/Devanagari text without destroying Hindi characters."""
    text = unicodedata.normalize("NFKC", text or "").strip().lower()
    for old, new in _NORMALIZATION_REPLACEMENTS.items():
        text = text.replace(old, new)

    # Keep Unicode word characters (including Devanagari), digits, and spaces.
    text = re.sub(r"[^\w\s-]", " ", text, flags=re.UNICODE)
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_gazetteer() -> List[dict]:
    rows: List[dict] = []
    with GAZETTEER_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            names = [row["locality"]]
            names.extend(a.strip() for a in row.get("aliases", "").split("|") if a.strip())
            rows.append(
                {
                    "locality": row["locality"],
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "names": names,
                    "normalized_names": [_normalize(n) for n in names],
                }
            )
    return rows


_GAZETTEER = _load_gazetteer()


def _score(query: str, candidate: str) -> float:
    """Return a 0..1 fuzzy confidence for one normalized candidate string."""
    if not query or not candidate:
        return 0.0
    if query == candidate:
        return 1.0

    # A locality/landmark embedded in a longer complaint-style location string
    # is a strong signal, e.g. "opp sms hospital gate 2 near main road".
    if len(candidate) >= 3 and candidate in query:
        coverage = min(1.0, len(candidate) / max(1, len(query)))
        return min(0.99, 0.90 + 0.09 * coverage)
    if len(query) >= 3 and query in candidate:
        coverage = min(1.0, len(query) / max(1, len(candidate)))
        return min(0.97, 0.84 + 0.12 * coverage)

    scores = (
        fuzz.WRatio(query, candidate),
        fuzz.token_set_ratio(query, candidate),
        fuzz.token_sort_ratio(query, candidate),
        fuzz.ratio(query, candidate),
    )
    return max(scores) / 100.0


def _local_match(location_text: str) -> Tuple[Optional[float], Optional[float], Optional[str], float]:
    query = _normalize(location_text)
    if not query:
        return None, None, None, 0.0

    best_row = None
    best_score = 0.0

    for row in _GAZETTEER:
        row_score = max(_score(query, alias) for alias in row["normalized_names"])
        if row_score > best_score:
            best_score = row_score
            best_row = row

    if best_row is None:
        return None, None, None, 0.0

    return best_row["lat"], best_row["lon"], best_row["locality"], round(best_score, 4)


def _read_cache() -> Dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_cache(cache: Dict[str, dict]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(CACHE_PATH)


def _cached_nominatim(location_text: str) -> Optional[dict]:
    global _last_request_at

    normalized = _normalize(location_text)
    if not normalized:
        return None

    cache_key = normalized
    with _cache_lock:
        cache = _read_cache()
        if cache_key in cache:
            return cache[cache_key] or None

    params = {
        "q": f"{location_text}, Jaipur, Rajasthan, India",
        "format": "jsonv2",
        "limit": 1,
        "addressdetails": 1,
        "countrycodes": "in",
    }
    url = f"{NOMINATIM_URL}?{urlencode(params)}"

    try:
        with _rate_lock:
            elapsed = time.monotonic() - _last_request_at
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)

            req = Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "en,hi"})
            try:
                with urlopen(req, timeout=8) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            finally:
                _last_request_at = time.monotonic()

        result = payload[0] if isinstance(payload, list) and payload else None
    except (OSError, ValueError, json.JSONDecodeError):
        result = None

    # Cache misses too, preventing repeated failed network lookups.
    with _cache_lock:
        cache = _read_cache()
        cache[cache_key] = result
        _write_cache(cache)

    return result


def _nominatim_confidence(result: dict) -> float:
    """Derive a conservative 0..1 confidence from a Nominatim result."""
    display = str(result.get("display_name", "")).lower()
    address = result.get("address") or {}
    importance = float(result.get("importance") or 0.0)

    # The query is Jaipur-scoped; still verify the returned address looks local.
    haystack = " ".join(str(v).lower() for v in address.values()) + " " + display
    is_jaipur = "jaipur" in haystack or "जयपुर" in haystack
    is_rajasthan = "rajasthan" in haystack or "राजस्थान" in haystack

    if is_jaipur and is_rajasthan:
        return round(min(0.85, max(0.65, 0.65 + importance * 0.20)), 4)
    if is_jaipur:
        return round(min(0.72, max(0.60, 0.60 + importance * 0.15)), 4)
    return round(min(0.55, max(0.35, 0.35 + importance * 0.20)), 4)


def geocode(location_text: str) -> Tuple[Optional[float], Optional[float], Optional[str], float]:
    """Geocode a Jaipur location string.

    Returns:
        (lat, lon, matched_name, confidence)

    Strategy:
      1. Fuzzy-match canonical names and aliases in data/gazetteer.csv.
      2. If local confidence >= 0.60, return the gazetteer result.
      3. Otherwise query cached, rate-limited Nominatim.
      4. If Nominatim has no usable result, return the weak local match only
         when it is non-zero; otherwise return (None, None, None, 0).

    Note: the requirement says unmatched inputs should return the all-None tuple,
    so weak fuzzy matches are not surfaced after a failed fallback.
    """
    if not isinstance(location_text, str) or not location_text.strip():
        return None, None, None, 0.0

    lat, lon, matched_name, confidence = _local_match(location_text)
    if confidence >= FUZZY_THRESHOLD:
        return lat, lon, matched_name, confidence

    result = _cached_nominatim(location_text)
    if result:
        try:
            nom_lat = float(result["lat"])
            nom_lon = float(result["lon"])
        except (KeyError, TypeError, ValueError):
            return None, None, None, 0.0

        display_name = str(result.get("display_name") or location_text)
        return nom_lat, nom_lon, display_name, _nominatim_confidence(result)

    return None, None, None, 0.0


if __name__ == "__main__":
    examples = [
        "Mansarover sec 7",
        "near GT Malviya Nagar",
        "opp. SMS Hospital gate 2",
        "राजा पार्क",
        "जगतपुरा community park",
    ]
    for example in examples:
        print(example, "->", geocode(example))
