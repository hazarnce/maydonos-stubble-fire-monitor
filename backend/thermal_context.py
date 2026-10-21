"""OpenStreetMap context enrichment for Turkish FIRMS thermal detections.

This module enriches reported thermal hotspots with nearby mapped land-use
context. Context tags are independent of model scores and are never proof of
incident cause."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import requests
import streamlit as st

OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT = (4, 16)
MAX_CONTEXT_POINTS = 15
CONTEXT_RADIUS_M = 25_000


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi, d_lam = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    value = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(value)))


def _category(tags: dict[str, Any]) -> tuple[str, str] | None:
    if tags.get("landuse") == "quarry" or tags.get("mining") or tags.get("resource"):
        return "maden", str(tags.get("name") or "OSM maden / taş ocağı")
    if (
        tags.get("man_made") in {"petroleum_well", "oil_well", "flare", "gas_flare"}
        or tags.get("industrial") in {"oil", "gas", "refinery"}
        or tags.get("petroleum") == "refinery"
    ):
        return "petrol_gaz", str(tags.get("name") or "OSM petrol / gaz tesisi")
    if tags.get("power") in {"plant", "generator", "substation"}:
        return "enerji", str(tags.get("name") or "OSM enerji tesisi")
    if tags.get("industrial") or tags.get("landuse") == "industrial" or tags.get("man_made") in {"works", "factory"}:
        return "sanayi", str(tags.get("name") or "OSM sanayi alanı")
    if tags.get("natural") in {"wood", "scrub", "heath"} or tags.get("landuse") == "forest":
        return "orman_calilik", str(tags.get("name") or "OSM orman / çalılık")
    if tags.get("landuse") in {"farmland", "farm", "orchard", "vineyard", "meadow", "grass"}:
        return "tarim", str(tags.get("name") or "OSM tarım / otlak alanı")
    if tags.get("natural") in {"water", "wetland"} or tags.get("landuse") in {"reservoir", "basin"}:
        return "su", str(tags.get("name") or "OSM su alanı")
    if tags.get("landuse") in {"residential", "commercial", "retail"}:
        return "yerlesim", str(tags.get("name") or "OSM yerleşim alanı")
    return None


@st.cache_data(ttl=6 * 60 * 60, max_entries=32, show_spinner=False)
def _fetch_osm_elements(points: tuple[tuple[float, float], ...]) -> tuple[dict[str, Any], ...]:
    clauses: list[str] = []
    for lat, lon in points:
        center = f"{lat:.5f},{lon:.5f}"
        radius = CONTEXT_RADIUS_M
        clauses.extend([
            f'nwr(around:{radius},{center})["industrial"];',
            f'nwr(around:{radius},{center})["landuse"~"industrial|quarry|forest|farmland|farm|orchard|vineyard|meadow|grass|residential|commercial|retail|reservoir|basin"];',
            f'nwr(around:{radius},{center})["power"~"plant|generator|substation"];',
            f'nwr(around:{radius},{center})["man_made"~"works|factory|petroleum_well|oil_well|flare|gas_flare"];',
            f'nwr(around:{radius},{center})["natural"~"wood|scrub|heath|water|wetland"];',
            f'nwr(around:{radius},{center})["mining"];',
        ])
    query = "[out:json][timeout:15];(" + "".join(clauses) + ");out center tags;"
    response = requests.post(
        OVERPASS_ENDPOINT,
        data={"data": query},
        headers={"User-Agent": "MaydonOs-Turkiye/1.0 (thermal context; OSM attribution in app)"},
        timeout=OVERPASS_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    elements = payload.get("elements", [])
    return tuple(element for element in elements if isinstance(element, dict))


def _coordinates(element: dict[str, Any]) -> tuple[float, float] | None:
    if isinstance(element.get("lat"), (int, float)) and isinstance(element.get("lon"), (int, float)):
        return float(element["lat"]), float(element["lon"])
    center = element.get("center")
    if isinstance(center, dict) and isinstance(center.get("lat"), (int, float)) and isinstance(center.get("lon"), (int, float)):
        return float(center["lat"]), float(center["lon"])
    return None


def _candidate(nearest: dict[str, tuple[str, float]] | None) -> str:
    if not nearest:
        return "Bağlam yetersiz / belirsiz"
    if "petrol_gaz" in nearest and nearest["petrol_gaz"][1] <= 1500:
        return "Petrol / gaz kaynağı yakınında termal olay adayı"
    if "maden" in nearest and nearest["maden"][1] <= 1500:
        return "Maden sahası yakınında termal olay adayı"
    if "sanayi" in nearest and nearest["sanayi"][1] <= 1500:
        return "Sanayi alanı yakınında termal olay adayı"
    if "enerji" in nearest and nearest["enerji"][1] <= 1500:
        return "Enerji tesisi yakınında termal olay adayı"
    if "orman_calilik" in nearest and nearest["orman_calilik"][1] <= 2500:
        return "Orman / çalılık çevresinde yangın adayı"
    if "tarim" in nearest and nearest["tarim"][1] <= 1500:
        return "Tarım alanı çevresinde yanma adayı"
    return "Kaynak belirsiz"


def enrich_hotspots_with_osm(hotspots: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Add nearest OSM context and conservative candidate labels to live points."""
    result = hotspots.copy()
    result["osm_candidate"] = "Bağlam yetersiz / belirsiz"
    result["osm_nearby"] = ""
    result["osm_distance_m"] = pd.NA
    if result.empty or not {"latitude", "longitude"}.issubset(result.columns):
        return result, "Konum bağlamı için gerçek FIRMS noktası yok."

    real = result
    if "is_demo" in real.columns:
        real = real.loc[~real["is_demo"].fillna(False).astype(bool)]
    if real.empty:
        return result, "Sentetik demo kayıtları için OpenStreetMap bağlamı sorgulanmadı."

    ranked = real.copy()
    ranked["_frp"] = pd.to_numeric(ranked.get("frp", pd.Series(index=ranked.index, dtype=float)), errors="coerce")
    ranked = ranked.sort_values("_frp", ascending=False, na_position="last").head(MAX_CONTEXT_POINTS)
    query_points = tuple(sorted({(round(float(row.latitude), 4), round(float(row.longitude), 4)) for row in ranked.itertuples()}))
    try:
        elements = _fetch_osm_elements(query_points)
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        kind = type(exc).__name__
        if kind == "Timeout":
            return result, "OpenStreetMap bağlamı zaman aşımına uğradı; termal olay sınıfı belirsiz bırakıldı."
        return result, "OpenStreetMap bağlamı alınamadı; termal olay sınıfı belirsiz bırakıldı."

    features: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for element in elements:
        tags = element.get("tags") if isinstance(element.get("tags"), dict) else {}
        categorized = _category(tags)
        coords = _coordinates(element)
        if not categorized or not coords:
            continue
        identity = (str(element.get("type", "")), int(element.get("id", -1)))
        if identity in seen:
            continue
        seen.add(identity)
        features.append({
            "category": categorized[0], "name": categorized[1],
            "latitude": coords[0], "longitude": coords[1],
        })

    for index, row in ranked.iterrows():
        lat, lon = float(row["latitude"]), float(row["longitude"])
        nearest: dict[str, tuple[str, float]] = {}
        for feature in features:
            distance = haversine_m(lat, lon, feature["latitude"], feature["longitude"])
            if distance > CONTEXT_RADIUS_M:
                continue
            category = str(feature["category"])
            if category not in nearest or distance < nearest[category][1]:
                nearest[category] = (str(feature["name"]), distance)
        if not nearest:
            result.at[index, "osm_candidate"] = "Yakında eşleşen OSM kaydı bulunmadı"
            result.at[index, "osm_nearby"] = ""
            continue
        closest_category, (closest_name, closest_distance) = min(nearest.items(), key=lambda item: item[1][1])
        context_text = "; ".join(
            f"{category}: {name} ({distance / 1000:.1f} km)"
            for category, (name, distance) in sorted(nearest.items(), key=lambda item: item[1][1])[:3]
        )
        result.at[index, "osm_candidate"] = _candidate(nearest)
        result.at[index, "osm_nearby"] = context_text
        result.at[index, "osm_distance_m"] = closest_distance

    return result, f"OpenStreetMap • {len(features)} eşleşen yakın çevre kaydı; en fazla {MAX_CONTEXT_POINTS} sıcak nokta bağlamlandırıldı"
