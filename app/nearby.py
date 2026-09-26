"""Find nearby dental facilities matched to the screening result.

OpenStreetMap (Overpass) is the default — no API key, no card.
If GOOGLE_PLACES_API_KEY is set, Google Places is used instead so we can
rank by public ratings.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import httpx

from .config import settings

logger = logging.getLogger("cariospectra.nearby")

_OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

_DENTAL_NAME = re.compile(
    r"dent|oral\s*(surg|care|health)|odont|tooth|cavity|endodont",
    re.I,
)
_EMERGENCY_NAME = re.compile(r"emerg|24\s*/*\s*7|trauma|oral\s*surg|hospital", re.I)
_SPECIALITY_DENTAL = re.compile(r"dent|oral|odont", re.I)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


def _coords(el: dict) -> Optional[Tuple[float, float]]:
    if "lat" in el and "lon" in el:
        return float(el["lat"]), float(el["lon"])
    c = el.get("center") or {}
    if "lat" in c and "lon" in c:
        return float(c["lat"]), float(c["lon"])
    return None


def _place_kind(tags: dict, name: str) -> str:
    amenity = (tags.get("amenity") or "").lower()
    healthcare = (tags.get("healthcare") or "").lower()
    speciality = (
        tags.get("healthcare:speciality")
        or tags.get("healthcare:specialty")
        or ""
    )
    if amenity == "dentist" or healthcare == "dentist":
        return "dentist"
    if _SPECIALITY_DENTAL.search(speciality) or _DENTAL_NAME.search(name):
        if amenity == "hospital" or healthcare == "hospital":
            return "dental hospital"
        return "dental clinic"
    if amenity == "hospital" or healthcare == "hospital":
        return "hospital"
    if amenity == "clinic" or healthcare == "clinic":
        return "clinic"
    return "dental care"


def _treats_issue(kind: str, name: str, tags: dict, include_emergency: bool) -> bool:
    """Only keep places that can actually treat caries / routine dental care."""
    if kind in {"dentist", "dental clinic", "dental hospital"}:
        return True
    if include_emergency and kind == "hospital":
        speciality = (
            tags.get("healthcare:speciality")
            or tags.get("healthcare:specialty")
            or ""
        )
        return bool(
            _SPECIALITY_DENTAL.search(speciality)
            or _DENTAL_NAME.search(name)
            or tags.get("emergency") == "yes"
        )
    return False


def _why(kind: str, include_emergency: bool, level: str) -> str:
    if kind == "dentist":
        if level == "routine":
            return "General dentist — routine exam and hygiene"
        return "General dentist — diagnoses and treats cavities"
    if kind == "dental clinic":
        return "Dental clinic — restorative / cavity care"
    if kind == "dental hospital":
        return "Dental hospital — can handle more advanced caries care"
    if include_emergency:
        return "Hospital with emergency or dental services — for pain, swelling, or infection"
    return "Dental facility"


def _overpass_query(lat: float, lng: float, radius_m: int, include_emergency: bool) -> str:
    parts = [
        f'nwr["amenity"="dentist"](around:{radius_m},{lat},{lng});',
        f'nwr["healthcare"="dentist"](around:{radius_m},{lat},{lng});',
        f'nwr["healthcare:speciality"~"dent",i](around:{radius_m},{lat},{lng});',
    ]
    if include_emergency:
        parts.append(
            f'nwr["amenity"="hospital"]["healthcare:speciality"~"dent",i](around:{radius_m},{lat},{lng});'
        )
        parts.append(
            f'nwr["amenity"="hospital"]["name"~"dent|oral",i](around:{radius_m},{lat},{lng});'
        )
    joined = "\n  ".join(parts)
    return f"[out:json][timeout:20];\n(\n  {joined}\n);\nout center tags;"


async def _fetch_overpass(query: str) -> List[dict]:
    async with httpx.AsyncClient(timeout=25.0) as client:
        last_err: Optional[Exception] = None
        for url in _OVERPASS_URLS:
            try:
                resp = await client.post(url, data={"data": query})
                if resp.status_code >= 400:
                    last_err = RuntimeError(f"Overpass {resp.status_code}")
                    continue
                return list(resp.json().get("elements") or [])
            except Exception as exc:
                last_err = exc
                logger.warning("Overpass failed at %s: %s", url, exc)
        if last_err:
            raise RuntimeError(f"Could not reach map data: {last_err}") from last_err
    return []


def _from_overpass(
    elements: List[dict],
    lat: float,
    lng: float,
    include_emergency: bool,
    level: str,
) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    places: List[Dict[str, Any]] = []
    for el in elements:
        tags = el.get("tags") or {}
        name = (tags.get("name") or tags.get("name:en") or "").strip()
        if not name:
            continue
        coords = _coords(el)
        if not coords:
            continue
        plat, plng = coords
        kind = _place_kind(tags, name)
        if not _treats_issue(kind, name, tags, include_emergency):
            continue
        key = f"{name.lower()}|{round(plat, 4)}|{round(plng, 4)}"
        if key in seen:
            continue
        seen.add(key)
        dist = haversine_km(lat, lng, plat, plng)
        rating = None
        try:
            if tags.get("stars"):
                rating = float(tags["stars"])
        except (TypeError, ValueError):
            rating = None
        emergency = bool(_EMERGENCY_NAME.search(name) or tags.get("emergency") == "yes")
        places.append(
            {
                "id": str(el.get("id", key)),
                "name": name,
                "kind": kind,
                "lat": plat,
                "lng": plng,
                "distance_km": round(dist, 2),
                "rating": rating,
                "reviews": None,
                "phone": tags.get("phone") or tags.get("contact:phone"),
                "address": tags.get("addr:full")
                or ", ".join(
                    p
                    for p in (
                        tags.get("addr:street"),
                        tags.get("addr:city"),
                    )
                    if p
                )
                or None,
                "open_now": None,
                "why": _why(kind, include_emergency, level),
                "emergency": emergency,
            }
        )
    return places


async def _fetch_google(
    lat: float, lng: float, radius_m: int, include_emergency: bool, level: str
) -> List[Dict[str, Any]]:
    key = settings.google_places_api_key
    if not key:
        return []
    queries = [("dentist", "dentist")]
    if include_emergency:
        queries.append(("emergency dentist", None))
    seen: set[str] = set()
    places: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=20.0) as client:
        for keyword, ptype in queries:
            params = {
                "location": f"{lat},{lng}",
                "radius": str(radius_m),
                "key": key,
            }
            if ptype:
                params["type"] = ptype
            if keyword:
                params["keyword"] = keyword
            resp = await client.get(
                "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
                params=params,
            )
            if resp.status_code >= 400:
                logger.warning("Google Places HTTP %s", resp.status_code)
                continue
            data = resp.json()
            if data.get("status") not in {"OK", "ZERO_RESULTS"}:
                logger.warning("Google Places status %s", data.get("status"))
                continue
            for el in data.get("results") or []:
                types = set(el.get("types") or [])
                name = (el.get("name") or "").strip()
                if not name:
                    continue
                loc = (el.get("geometry") or {}).get("location") or {}
                if "lat" not in loc:
                    continue
                plat, plng = float(loc["lat"]), float(loc["lng"])
                pid = el.get("place_id") or f"{name}|{plat}"
                if pid in seen:
                    continue
                dental = bool(
                    types & {"dentist", "dental_clinic"}
                    or _DENTAL_NAME.search(name)
                )
                hospital = "hospital" in types
                if not dental and not (include_emergency and hospital):
                    continue
                if hospital and not dental:
                    kind = "hospital"
                    if not include_emergency:
                        continue
                elif "hospital" in types:
                    kind = "dental hospital"
                else:
                    kind = "dentist"
                seen.add(pid)
                dist = haversine_km(lat, lng, plat, plng)
                rating = el.get("rating")
                places.append(
                    {
                        "id": pid,
                        "name": name,
                        "kind": kind,
                        "lat": plat,
                        "lng": plng,
                        "distance_km": round(dist, 2),
                        "rating": float(rating) if rating is not None else None,
                        "reviews": el.get("user_ratings_total"),
                        "phone": None,
                        "address": el.get("vicinity"),
                        "open_now": (el.get("opening_hours") or {}).get("open_now"),
                        "why": _why(kind, include_emergency, level),
                        "emergency": bool(include_emergency and (hospital or _EMERGENCY_NAME.search(name))),
                    }
                )
    return places


def _rank(places: List[Dict[str, Any]], include_emergency: bool) -> List[Dict[str, Any]]:
    def score(p: dict) -> float:
        dist = max(0.15, float(p["distance_km"]))
        rating = float(p["rating"]) if p.get("rating") is not None else 3.6
        reviews = float(p["reviews"] or 0)
        review_boost = math.log10(reviews + 10) / 3.0
        emergency_boost = 0.35 if include_emergency and p.get("emergency") else 0.0
        kind_boost = {
            "dentist": 0.4,
            "dental clinic": 0.35,
            "dental hospital": 0.3,
            "hospital": 0.05,
        }.get(p.get("kind"), 0.1)
        # Higher is better: rating and fit, divided by distance.
        return (rating + review_boost + emergency_boost + kind_boost) / (0.6 + dist)

    return sorted(places, key=score, reverse=True)


async def find_nearby_care(
    lat: float,
    lng: float,
    level: str = "moderate",
    limit: int = 8,
) -> Dict[str, Any]:
    include_emergency = level == "high"
    radius_m = 12000 if include_emergency else 8000
    source = "openstreetmap"
    places: List[Dict[str, Any]] = []

    if settings.google_places_api_key:
        try:
            places = await _fetch_google(lat, lng, radius_m, include_emergency, level)
            if places:
                source = "google"
        except Exception:
            logger.exception("Google Places failed; falling back to OpenStreetMap")

    if not places:
        elements = await _fetch_overpass(
            _overpass_query(lat, lng, radius_m, include_emergency)
        )
        places = _from_overpass(elements, lat, lng, include_emergency, level)

    ranked = _rank(places, include_emergency)[: max(1, min(limit, 12))]
    return {"source": source, "places": ranked, "include_emergency": include_emergency}
