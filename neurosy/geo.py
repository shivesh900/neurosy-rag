"""Find and rank nearby healthcare providers.

Uses free OpenStreetMap services (no API key): Nominatim for geocoding and the
Overpass API for hospitals, clinics and doctors. Results are ranked with a
weighted score, so a high-risk condition favours a hospital with an emergency
department, and a skin rash favours a nearby clinic with a matching speciality.

    score = 0.40 * proximity   (exp(-distance / 4 km))
          + 0.25 * specialty match
          + 0.15 * emergency readiness (weighted by the condition's risk)
          + 0.10 * facility fit (hospital vs clinic for the risk level)
          + 0.10 * data quality (opening hours / phone / website known)

If the network is unavailable, a bundled OpenStreetMap snapshot around SRM
Kattankulathur (Chennai) is used. Map data (c) OpenStreetMap contributors, ODbL.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import requests

from .knowledge import DATA_DIR

UA = {"User-Agent": "NeuroSy-RAG/1.0 (educational project; github.com/shivesh900/neurosy-rag)"}
OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
NOMINATIM = "https://nominatim.openstreetmap.org/search"
SNAPSHOT = DATA_DIR / "osm_providers_chennai_srm.json"

# knowledge-base specialty -> OSM healthcare:speciality / name keywords
SPECIALTY_TERMS = {
    "Infectious Diseases": ["infectious", "general", "internal", "fever"],
    "General Medicine": ["general", "internal", "family"],
    "Pulmonology": ["pulmonology", "chest", "respiratory", "tb"],
    "ENT": ["otolaryngology", "ent", "ear"],
    "Ophthalmology": ["ophthalmology", "eye"],
    "Gastroenterology": ["gastroenterology", "gastro", "digestive", "liver"],
    "General Surgery": ["surgery", "general"],
    "Urology": ["urology", "kidney", "nephrology"],
    "Cardiology": ["cardiology", "heart", "cardiac"],
    "Endocrinology": ["endocrinology", "diabetes", "thyroid"],
    "Hematology": ["haematology", "hematology", "blood"],
    "Neurology": ["neurology", "neuro"],
    "Rheumatology": ["rheumatology", "arthritis", "ortho"],
    "Orthopedics": ["orthopaedics", "orthopedics", "ortho", "physiotherapy", "spine"],
    "Dermatology": ["dermatology", "skin"],
    "Psychiatry": ["psychiatry", "mental", "counselling"],
}
RISK_EMERGENCY_WEIGHT = {"high": 1.0, "medium": 0.5, "low": 0.15}


@dataclass
class Provider:
    name: str
    lat: float
    lon: float
    kind: str
    distance_km: float = 0.0
    emergency: bool = False
    speciality: str = ""
    phone: bool = False
    website: bool = False
    opening_hours: str = ""
    score: float = 0.0
    why: str = ""

    def to_dict(self):
        d = asdict(self)
        d["distance_km"] = round(d["distance_km"], 2)
        d["score"] = round(d["score"], 3)
        d["maps_url"] = f"https://www.openstreetmap.org/?mlat={self.lat}&mlon={self.lon}#map=17/{self.lat}/{self.lon}"
        return d


def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def geocode(place: str, timeout: float = 10) -> Optional[Tuple[float, float, str]]:
    r = requests.get(NOMINATIM, params={"q": place, "format": "json", "limit": 1}, headers=UA, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    if not js:
        return None
    return float(js[0]["lat"]), float(js[0]["lon"]), js[0]["display_name"]


def _parse_elements(elements) -> List[Provider]:
    out = []
    for el in elements:
        tags = el.get("tags", {})
        lat = el.get("lat") or el.get("center", {}).get("lat")
        lon = el.get("lon") or el.get("center", {}).get("lon")
        name = tags.get("name") or tags.get("name:en")
        if lat is None or not name:
            continue
        kind = tags.get("amenity") or tags.get("healthcare") or "clinic"
        out.append(Provider(
            name=name, lat=float(lat), lon=float(lon), kind=kind,
            emergency=tags.get("emergency") == "yes",
            speciality=tags.get("healthcare:speciality", ""),
            phone=bool(tags.get("phone") or tags.get("contact:phone")),
            website=bool(tags.get("website") or tags.get("contact:website")),
            opening_hours=tags.get("opening_hours", ""),
        ))
    return out


def fetch_providers(lat: float, lon: float, radius_m: int = 6000, timeout: float = 25) -> List[Provider]:
    q = f"""[out:json][timeout:{int(timeout)}];
(
  nwr["amenity"~"^(hospital|clinic|doctors)$"](around:{radius_m},{lat},{lon});
  nwr["healthcare"~"^(hospital|clinic|doctor)$"](around:{radius_m},{lat},{lon});
);
out center tags;"""
    last = None
    for url in OVERPASS_MIRRORS:  # public Overpass servers are often busy; try mirrors in turn
        try:
            r = requests.post(url, data={"data": q}, headers=UA, timeout=timeout + 5)
            r.raise_for_status()
            return _parse_elements(r.json().get("elements", []))
        except Exception as exc:
            last = exc
    raise RuntimeError(f"all Overpass mirrors failed: {last}")


def load_snapshot() -> Tuple[Tuple[float, float], List[Provider]]:
    js = json.loads(Path(SNAPSHOT).read_text())
    return tuple(js["center"]), _parse_elements(js["elements"])


def rank_providers(providers: List[Provider], origin: Tuple[float, float], specialty: str = "General Medicine",
                   risk_level: str = "low", top_n: int = 5) -> List[Provider]:
    terms = SPECIALTY_TERMS.get(specialty, ["general"])
    ew = RISK_EMERGENCY_WEIGHT.get(risk_level, 0.3)
    seen, ranked = set(), []
    for p in providers:
        key = (p.name.lower(), round(p.lat, 3), round(p.lon, 3))
        if key in seen:
            continue
        seen.add(key)
        p.distance_km = haversine_km(origin, (p.lat, p.lon))
        proximity = math.exp(-p.distance_km / 4.0)
        hay = f"{p.speciality} {p.name}".lower()
        spec = 1.0 if any(t in hay for t in terms) else (0.5 if p.kind == "hospital" else 0.2)
        emerg = (1.0 if p.emergency else (0.4 if p.kind == "hospital" else 0.0)) * ew + (1 - ew) * 0.5
        fit = (1.0 if p.kind == "hospital" else 0.4) if risk_level == "high" else (0.8 if p.kind != "hospital" else 0.7)
        quality = (bool(p.opening_hours) + p.phone + p.website) / 3
        p.score = 0.40 * proximity + 0.25 * spec + 0.15 * emerg + 0.10 * fit + 0.10 * quality
        why = [f"{p.distance_km:.1f} km away"]
        if spec == 1.0:
            why.append(f"matches {specialty}")
        if p.emergency:
            why.append("has emergency department")
        why.append(p.kind)
        p.why = ", ".join(why)
        ranked.append(p)
    ranked.sort(key=lambda p: p.score, reverse=True)
    return ranked[:top_n]


def find_care(place: Optional[str] = None, lat: Optional[float] = None, lon: Optional[float] = None,
              specialty: str = "General Medicine", risk_level: str = "low", top_n: int = 5,
              offline: bool = False) -> dict:
    """Locate the user and return ranked providers. Falls back to the bundled snapshot offline."""
    source, label = "openstreetmap-live", place or (f"{lat:.4f}, {lon:.4f}" if lat is not None else None)
    try:
        if offline:
            raise RuntimeError("offline mode")
        if lat is None and place:
            g = geocode(place)
            if not g:
                return {"error": f"Could not find '{place}'.", "providers": []}
            lat, lon, label = g
        if lat is None:
            raise RuntimeError("no location given")
        providers = fetch_providers(lat, lon)
        if not providers:
            providers = fetch_providers(lat, lon, radius_m=15000)
    except Exception as exc:  # network down, rate limited, no location...
        (lat, lon), providers = load_snapshot()
        source = f"bundled OSM snapshot (SRM Kattankulathur, Chennai), because: {exc}"
        label = "SRM Kattankulathur, Chennai (sample)"
    ranked = rank_providers(providers, (lat, lon), specialty, risk_level, top_n)
    return {"origin": {"lat": lat, "lon": lon, "label": label}, "source": source,
            "attribution": "Map data (c) OpenStreetMap contributors, ODbL",
            "providers": [p.to_dict() for p in ranked]}
