"""Public web interface API for MaydonOs.

Run locally with: python -m uvicorn backend.web_api:app --reload
The API key is read only from server-side environment / Streamlit secrets.
"""

from __future__ import annotations

import math
import os
import json
import logging
import unicodedata
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import truststore

    truststore.inject_into_ssl()
except (ImportError, AttributeError):
    pass

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.corine_context import (
    classify_hotspot_tr,
    enrich_hotspots_with_corine,
    enrich_hotspots_with_local_raster,
)
from ml.live_model import score_live_events
from ml.harvest_calendar import (
    HARVEST_SOURCES,
    harvest_context as build_harvest_context,
    regional_calendar_payload,
)
from backend.web_history import MIN_HISTORY_DAYS, _area_id, read_history, record_snapshot, score_snapshot
from backend.engine import (
    calculate_cffdrs_system,
    classify_ffmc,
    fetch_nasa_firms_hotspots,
    fetch_nasa_firms_extent_hotspots,
    fetch_open_meteo_weather,
    anomaly_priority,
    previous_24h_rain,
    secret_or_environment,
    validate_coordinates,
)

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "frontend"
BOUNDARY_ROOT = WEB_ROOT / "assets" / "turkey-boundaries"


@lru_cache(maxsize=1)
def _province_geojson() -> dict[str, Any]:
    path = BOUNDARY_ROOT / "provinces.geojson"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") != "FeatureCollection" or len(payload.get("features", [])) != 81:
        raise RuntimeError("Türkiye il sınırları dosyası eksik veya geçersiz.")
    return payload


@lru_cache(maxsize=81)
def _district_geojson(province_code: str) -> dict[str, Any]:
    if province_code not in {f["properties"]["province_code"] for f in _province_geojson()["features"]}:
        raise HTTPException(status_code=400, detail="Geçersiz il seçildi.")
    path = BOUNDARY_ROOT / f"{province_code}.geojson"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") != "FeatureCollection":
        raise RuntimeError(f"{province_code} ilçe sınırları dosyası geçersiz.")
    return payload


def _normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold().replace("ı", "i"))
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _district_feature(province_code: str, name: str) -> dict[str, Any]:
    target = _normalize_name(name)
    for feature in _district_geojson(province_code)["features"]:
        properties = feature.get("properties", {})
        if _normalize_name(str(properties.get("shapeName", properties.get("name", "")))) == target:
            return feature
    raise HTTPException(status_code=400, detail="Bu ilçe için OSM sınırı bulunamadı.")


@lru_cache(maxsize=81)
def _province_feature(province_code: str) -> dict[str, Any]:
    for feature in _province_geojson()["features"]:
        if feature.get("properties", {}).get("province_code") == province_code:
            return feature
    raise HTTPException(status_code=400, detail="Geçersiz il seçildi.")


@lru_cache(maxsize=1)
def _province_shapes() -> tuple[tuple[str, str, dict[str, Any], tuple[float, float, float, float]], ...]:
    shapes = []
    for feature in _province_geojson()["features"]:
        props = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        shapes.append((
            str(props.get("name", "")), str(props.get("province_code", "")),
            geometry, _geometry_bounds(geometry),
        ))
    return tuple(shapes)


@lru_cache(maxsize=1)
def _province_spatial_index() -> tuple[Any, tuple[Any, ...], tuple[Any, ...]] | None:
    """Build a cached R-tree over province boundaries when Shapely is installed.

    STRtree returns bounding-box candidates; callers still apply an exact
    geometry predicate, so nearby provinces cannot become false matches.
    """
    try:
        from shapely.geometry import shape
        from shapely.strtree import STRtree

        rows = _province_shapes()
        geometries = tuple(shape(row[2]) for row in rows)
        return STRtree(geometries), geometries, rows
    except (ImportError, ValueError, TypeError):
        return None


@lru_cache(maxsize=100_000)
def _province_name_at(lon: float, lat: float) -> str | None:
    spatial_index = _province_spatial_index()
    if spatial_index is not None:
        try:
            from shapely.geometry import Point

            tree, geometries, rows = spatial_index
            point = Point(lon, lat)
            # Shapely 2 returns integer indices in input geometry order.
            for candidate in tree.query(point):
                index = int(candidate)
                if geometries[index].covers(point):
                    return rows[index][0] or None
            return None
        except (ImportError, IndexError, TypeError, ValueError):
            # Keep the existing GeoJSON implementation as a compatibility
            # fallback if the optional spatial-index path is unavailable.
            pass

    for name, _, geometry, bounds in _province_shapes():
        if _point_in_geometry(lon, lat, geometry, bounds):
            return name or None
    return None


def _turkey_bounds() -> tuple[float, float, float, float]:
    """Return a bounded extent derived from the shipped province boundaries."""
    bounds = [shape[3] for shape in _province_shapes()]
    return (
        min(item[0] for item in bounds), min(item[1] for item in bounds),
        max(item[2] for item in bounds), max(item[3] for item in bounds),
    )


def _harvest_context(observation_date: Any, province_name: str | None) -> dict[str, Any]:
    """Return the shared region-aware cereal-season context."""
    return build_harvest_context(observation_date, province_name)


def _event_identity(row: Any) -> tuple[float, float, str, str, str] | None:
    """Stable key for copying model results between map and detail records."""
    try:
        lat = round(float(row.get("latitude")), 5)
        lon = round(float(row.get("longitude")), 5)
    except (TypeError, ValueError):
        return None
    day = str(row.get("acq_date") or "")
    minute = pd.to_numeric(row.get("acq_time"), errors="coerce")
    time_key = str(int(minute)) if pd.notna(minute) else ""
    sensor = str(row.get("satellite") or "")
    if not day or not sensor:
        return None
    return lat, lon, day, time_key, sensor


def _geometry_points(geometry: dict[str, Any]) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    def walk(value: Any) -> None:
        if isinstance(value, list) and len(value) >= 2 and all(isinstance(v, (int, float)) for v in value[:2]):
            lon, lat = float(value[0]), float(value[1])
            if math.isfinite(lon) and math.isfinite(lat):
                points.append((lon, lat))
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(geometry.get("coordinates", []))
    return points


def _geometry_bounds(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    points = _geometry_points(geometry)
    if not points:
        raise RuntimeError("OSM ilçe sınırı koordinat içermiyor.")
    lons, lats = zip(*points)
    return min(lons), min(lats), max(lons), max(lats)


def _ring_contains(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    if len(ring) < 4:
        return False
    xj, yj = ring[-1][0], ring[-1][1]
    for point in ring:
        xi, yi = point[0], point[1]
        if ((yi > lat) != (yj > lat)) and lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-20) + xi:
            inside = not inside
        xj, yj = xi, yi
    return inside


def _point_in_geometry(
    lon: float,
    lat: float,
    geometry: dict[str, Any],
    bounds: tuple[float, float, float, float] | None = None,
) -> bool:
    """Point-in-polygon for OSM Polygon/MultiPolygon GeoJSON including holes."""
    # The bounding-box rejection keeps exact polygon checks cheap when FIRMS
    # returns detections elsewhere in the district's rectangular extent.
    if bounds is not None:
        west, south, east, north = bounds
        if lon < west or lon > east or lat < south or lat > north:
            return False
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    polygons = [coordinates] if kind == "Polygon" else coordinates if kind == "MultiPolygon" else []
    for polygon in polygons:
        if not polygon or not _ring_contains(lon, lat, polygon[0]):
            continue
        if not any(_ring_contains(lon, lat, hole) for hole in polygon[1:]):
            return True
    return False

app = FastAPI(
    title="MaydonOs API",
    description="Açılışta Türkiye geneli NASA FIRMS; ilçe seçilince hava ve saha bağlamı.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
)
app.mount("/assets", StaticFiles(directory=WEB_ROOT / "assets"), name="assets")


@app.get("/", include_in_schema=False)
def homepage() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "MaydonOs"}


def _recent_events(frame: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
    if frame.empty or not {"acq_date", "acq_time"}.issubset(frame.columns):
        return frame.iloc[0:0].copy()
    hhmm = pd.to_numeric(frame["acq_time"], errors="coerce").map(
        lambda value: f"{int(value):04d}" if pd.notna(value) else ""
    )
    stamps = pd.to_datetime(
        frame["acq_date"].astype(str) + " " + hhmm,
        format="%Y-%m-%d %H%M", errors="coerce", utc=True,
    )
    now = pd.Timestamp.now(tz="UTC")
    return frame.loc[stamps.between(now - pd.Timedelta(hours=hours), now)].copy().reset_index(drop=True)


def _firms_observation_time(row: pd.Series | dict[str, Any]) -> pd.Timestamp | None:
    """Parse a FIRMS acquisition timestamp as UTC without inventing a time."""
    day = row.get("acq_date")
    raw_time = pd.to_numeric(row.get("acq_time"), errors="coerce")
    if pd.isna(day) or pd.isna(raw_time):
        return None
    try:
        return pd.to_datetime(
            f"{str(day)} {int(raw_time):04d}",
            format="%Y-%m-%d %H%M", utc=True, errors="raise",
        )
    except (TypeError, ValueError):
        return None


def _nearby_observations(
    all_events: pd.DataFrame,
    selected: pd.Series,
    radius_km: float = 1.0,
) -> list[dict[str, Any]]:
    """Return actual FIRMS records at/after the selected detection, within 1 km.

    This is a nearby-pixel observation history, not a claim that every row is the
    same fire. FIRMS does not provide continuous hourly FRP measurements.
    """
    selected_time = _firms_observation_time(selected)
    cutoff_time = pd.Timestamp.now(tz="UTC")
    try:
        lat0, lon0 = float(selected["latitude"]), float(selected["longitude"])
    except (KeyError, TypeError, ValueError):
        return []
    if selected_time is None or not math.isfinite(lat0) or not math.isfinite(lon0):
        return []

    candidates: list[tuple[pd.Timestamp, pd.Series]] = []
    if not all_events.empty:
        hhmm = pd.to_numeric(all_events["acq_time"], errors="coerce").map(
            lambda value: f"{int(value):04d}" if pd.notna(value) else ""
        )
        timestamps = pd.to_datetime(
            all_events["acq_date"].astype(str) + " " + hhmm,
            format="%Y-%m-%d %H%M", errors="coerce", utc=True,
        )
        latitudes = pd.to_numeric(all_events["latitude"], errors="coerce").to_numpy(dtype=float)
        longitudes = pd.to_numeric(all_events["longitude"], errors="coerce").to_numpy(dtype=float)
        lat_radians = np.radians(latitudes)
        lon_radians = np.radians(longitudes)
        lat0_radians, lon0_radians = math.radians(lat0), math.radians(lon0)
        delta_lat = lat_radians - lat0_radians
        delta_lon = lon_radians - lon0_radians
        haversine = (
            np.sin(delta_lat / 2) ** 2
            + math.cos(lat0_radians) * np.cos(lat_radians) * np.sin(delta_lon / 2) ** 2
        )
        distances = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
        eligible = (
            timestamps.notna().to_numpy()
            & (timestamps >= selected_time).to_numpy()
            & (timestamps <= cutoff_time).to_numpy()
            & np.isfinite(latitudes) & np.isfinite(longitudes)
            & (distances <= radius_km)
        )
        for position in np.flatnonzero(eligible):
            candidates.append((timestamps.iloc[position], all_events.iloc[int(position)]))

    # Include the selected point even when upstream data has an unusual duplicate
    # representation, then remove exact duplicate observation records.
    if not any(
        stamp == selected_time
        and str(row.get("satellite", "")) == str(selected.get("satellite", ""))
        and abs(float(row.get("latitude", lat0)) - lat0) < 1e-7
        and abs(float(row.get("longitude", lon0)) - lon0) < 1e-7
        for stamp, row in candidates
    ):
        candidates.append((selected_time, selected))

    candidates.sort(key=lambda item: (item[0], str(item[1].get("satellite", ""))))
    observations: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for stamp, row in candidates:
        signature = (
            stamp.isoformat(), str(row.get("satellite", "")),
            str(row.get("latitude", "")), str(row.get("longitude", "")),
        )
        if signature in seen:
            continue
        seen.add(signature)
        score, label, color = anomaly_priority(row)
        observed_lat = math.radians(float(row.get("latitude", lat0)))
        observed_lon = math.radians(float(row.get("longitude", lon0)))
        lat0_radians, lon0_radians = math.radians(lat0), math.radians(lon0)
        haversine = (
            math.sin((observed_lat - lat0_radians) / 2) ** 2
            + math.cos(lat0_radians) * math.cos(observed_lat)
            * math.sin((observed_lon - lon0_radians) / 2) ** 2
        )
        distance_km = 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(haversine)))
        observations.append({
            "time": stamp.isoformat(),
            "acq_date": str(row.get("acq_date", "")),
            "acq_time": _json_value(row.get("acq_time")),
            "satellite": _json_value(row.get("satellite")),
            "frp": _json_value(row.get("frp")),
            "confidence": _json_value(row.get("confidence")),
            "bright_ti4": _json_value(row.get("bright_ti4")),
            "distance_km": round(distance_km, 3),
            "priority_score": score if score > 0 else None,
            "priority_label": label,
            "priority_color": color,
        })
    return observations


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value):
        return None
    return value


@app.get("/api/districts")
def districts(response: Response, province: str = Query(..., min_length=5, max_length=5)) -> list[dict[str, Any]]:
    response.headers["Cache-Control"] = "public, max-age=86400"
    payload = _district_geojson(province)
    return [
        {
            "name": feature["properties"]["shapeName"],
            "province_code": province,
            "province_name": feature["properties"]["province_name"],
            "latitude": feature["properties"]["latitude"],
            "longitude": feature["properties"]["longitude"],
        }
        for feature in payload["features"]
    ]


@app.get("/api/district-boundaries")
def district_boundaries(
    response: Response,
    province: str | None = Query(default=None, min_length=5, max_length=5),
) -> dict[str, Any]:
    response.headers["Cache-Control"] = "public, max-age=86400"
    return _province_geojson() if province is None else _district_geojson(province)


@lru_cache(maxsize=512)
def _reverse_geocode(lat: float, lon: float) -> str | None:
    """Resolve a clicked FIRMS coordinate to a nearby OpenStreetMap address."""
    response = requests.get(
        "https://nominatim.openstreetmap.org/reverse",
        params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 16, "addressdetails": 1},
        headers={"User-Agent": "MaydonOs-Diyarbakir-Fire-Watch/1.0 (public map interface)"},
        timeout=(5, 10),
    )
    response.raise_for_status()
    payload = response.json()
    address = payload.get("address", {})
    parts = [
        address.get("road") or address.get("footway") or address.get("hamlet") or address.get("village"),
        address.get("neighbourhood") or address.get("suburb"),
        address.get("town") or address.get("city") or address.get("county"),
        address.get("state"),
    ]
    cleaned = list(dict.fromkeys(part for part in parts if part))
    return ", ".join(cleaned) or payload.get("display_name")


@app.get("/api/reverse-geocode")
def reverse_geocode(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
) -> dict[str, str | None]:
    try:
        return {"address": _reverse_geocode(round(lat, 5), round(lon, 5))}
    except (requests.RequestException, ValueError, TypeError):
        return {"address": None}


@lru_cache(maxsize=1024)
def _make_hotspot_note(
    acq_date: str,
    acq_time: str,
    frp: float | None,
    confidence: str,
    daynight: str,
    land_cover: str,
    ffmc: float | None,
) -> str:
    api_key = secret_or_environment("VISION_API_KEY")
    if not api_key or api_key.startswith("paste-"):
        raise RuntimeError("Otomatik not servisi yapılandırılmamış.")
    facts = {
        "tespit_tarihi": acq_date or "bilinmiyor",
        "tespit_saati_utc": acq_time or "bilinmiyor",
        "uydu_guveni": confidence or "bilinmiyor",
        "gece_gunduz": daynight or "bilinmiyor",
        "frp_mw": frp,
        "arazi_baglamı": land_cover or "bilinmiyor",
        "ffmc_gosterge": ffmc,
    }
    prompt = (
        "Bu NASA FIRMS sıcak nokta kaydı için Türkçe, en fazla 2 cümlelik kısa bir Sistem notu yaz. "
        "Yalnızca verilen alanları yorumla; yangın, anız yakımı, suç veya neden kesinleşti deme. "
        "FRP'yi ölçüm olarak açıkla, arazi sınıfının geniş ölçekli bağlam olduğunu belirt. "
        "Eksik veriden sonuç çıkarma; gerekiyorsa saha kontrolü gerektiğini söyle. "
        "Başlık veya model adı yazma. Sadece not metnini döndür.\n"
        f"Kayıt: {facts}"
    )
    response = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent",
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 120},
        },
        timeout=(5, 12),
    )
    response.raise_for_status()
    text = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    if not text or len(text) > 600:
        raise ValueError("Not servisi geçerli kısa bir metin döndürmedi.")
    return text


@app.post("/api/hotspot-note")
def hotspot_note(payload: dict[str, Any]) -> dict[str, str | None]:
    """Generate a cautious, cached note from FIRMS metadata only."""
    def clean_text(key: str, limit: int = 80) -> str:
        return str(payload.get(key) or "")[:limit]

    def optional_number(key: str, low: float, high: float) -> float | None:
        value = payload.get(key)
        if value is None or value == "":
            return None
        parsed = float(value)
        if not math.isfinite(parsed) or not low <= parsed <= high:
            raise ValueError(f"{key} aralık dışında.")
        return round(parsed, 2)

    try:
        note = _make_hotspot_note(
            clean_text("acq_date"), clean_text("acq_time", 8),
            optional_number("frp", 0, 10000), clean_text("confidence", 20),
            clean_text("daynight", 8), clean_text("land_cover", 100),
            optional_number("ffmc", 0, 100),
        )
        return {"note": note, "method": "online", "error": None}
    except Exception:
        try:
            frp = optional_number("frp", 0, 10000)
        except (ValueError, TypeError):
            frp = None
        try:
            ffmc = optional_number("ffmc", 0, 100)
        except (ValueError, TypeError):
            ffmc = None
        confidence = clean_text("confidence", 20).lower()
        confidence_label = {"h": "yüksek", "high": "yüksek", "n": "normal", "nominal": "normal", "l": "düşük", "low": "düşük"}.get(confidence, confidence or "belirtilmemiş")
        night = clean_text("daynight", 8).upper()
        time_label = "gece" if night == "N" else "gündüz" if night == "D" else "zaman bilgisi belirtilmemiş"
        land_cover = clean_text("land_cover", 100) or "bilinmiyor"
        parts = [f"Uydu noktası {time_label} ve {confidence_label} algılama güveniyle kaydedilmiş."]
        if frp is not None:
            parts.append(f"FRP {frp:g} MW, termal ışıma gücü ölçümüdür; tek başına yangını veya yanan alanı doğrulamaz.")
        else:
            parts.append("FRP değeri bulunmadığı için termal güç karşılaştırması yapılamıyor.")
        parts.append(f"Arazi bağlamı: {land_cover}.")
        if ffmc is not None:
            parts.append(f"Bölgesel FFMC göstergesi {ffmc:g}; bu hava temelli gösterge nokta bazında doğrulama değildir.")
        parts.append("Saha teyidi gerekir.")
        return {"note": " ".join(parts), "method": "measurements", "error": None}


@app.get("/api/nearby")
def nearby_hotspots(
    lat: float = Query(..., ge=35, le=43),
    lon: float = Query(..., ge=25, le=46),
    hours: int = Query(default=24, ge=6, le=96),
) -> dict[str, Any]:
    """Find recent FIRMS thermal observations within 10/25/50 km of a location."""
    validate_coordinates(lat, lon)
    # Request a slightly larger bounding box than the 50 km radius, then apply
    # exact great-circle distances below. The query stays under FIRMS' area cap.
    padded_radius_km = 55.0
    lat_delta = padded_radius_km / 111.32
    lon_delta = padded_radius_km / (111.32 * math.cos(math.radians(lat)))
    bbox = (
        max(25.0, lon - lon_delta), max(35.0, lat - lat_delta),
        min(46.0, lon + lon_delta), min(43.0, lat + lat_delta),
    )
    key = secret_or_environment("FIRMS_MAP_KEY")
    events, source_status = fetch_nasa_firms_hotspots(bbox, key, allow_demo=False)
    available = source_status.startswith("NASA FIRMS •")
    recent = _recent_events(events, hours=hours)
    observations: list[dict[str, Any]] = []
    counts = {"10": 0, "25": 0, "50": 0}
    nearby_total = 0
    if not recent.empty:
        latitudes = np.radians(pd.to_numeric(recent["latitude"], errors="coerce").to_numpy(dtype=float))
        longitudes = np.radians(pd.to_numeric(recent["longitude"], errors="coerce").to_numpy(dtype=float))
        lat0, lon0 = math.radians(lat), math.radians(lon)
        delta_lat, delta_lon = latitudes - lat0, longitudes - lon0
        haversine = np.sin(delta_lat / 2) ** 2 + math.cos(lat0) * np.cos(latitudes) * np.sin(delta_lon / 2) ** 2
        distances = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))
        recent = recent.assign(distance_km=distances)
        recent = recent.loc[recent["distance_km"] <= 50.0].sort_values("distance_km").reset_index(drop=True)
        nearby_total = len(recent)
        for radius in (10, 25, 50):
            counts[str(radius)] = int((recent["distance_km"] <= radius).sum())
        for _, row in recent.head(30).iterrows():
            score, label, color = anomaly_priority(row)
            observations.append({
                "latitude": _json_value(row.get("latitude")),
                "longitude": _json_value(row.get("longitude")),
                "distance_km": round(float(row["distance_km"]), 2),
                "acq_date": _json_value(row.get("acq_date")),
                "acq_time": _json_value(row.get("acq_time")),
                "satellite": _json_value(row.get("satellite")),
                "confidence": _json_value(row.get("confidence")),
                "frp": _json_value(row.get("frp")),
                "bright_ti4": _json_value(row.get("bright_ti4")),
                "priority_score": score if score > 0 else None,
                "priority_label": label,
                "priority_color": color,
            })
    return {
        "available": available,
        "source_status": source_status,
        "center": {"latitude": lat, "longitude": lon},
        "hours": hours,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "radius_km": 50,
        "counts": counts if available else None,
        "hotspots_total": nearby_total if available else None,
        "hotspots": observations if available else [],
        "notice": "Bunlar NASA FIRMS termal gözlemleridir; doğrulanmış yangın kaydı veya güvenlik mesafesi değildir.",
    }


@app.get("/api/dashboard")
def dashboard(
    district: str | None = Query(default=None, min_length=1, max_length=100),
    province: str | None = Query(default=None, min_length=5, max_length=5),
    hours: int = Query(default=24, ge=6, le=96),
    offset: int = Query(default=0, ge=0, le=20_000),
    day: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
) -> dict[str, Any]:
    if day:
        try:
            datetime.strptime(day, "%Y-%m-%d")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Gün biçimi YYYY-MM-DD olmalı.") from exc
    if district and not province:
        raise HTTPException(status_code=400, detail="İlçe sorgusu için il kodu gerekli.")
    scope = "district" if district else "province" if province else "country"
    feature: dict[str, Any] | None = None
    if scope == "district":
        feature = _district_feature(str(province), str(district))
    elif scope == "province":
        feature = _province_feature(str(province))

    if feature is not None:
        lat = float(feature["properties"]["latitude"])
        lon = float(feature["properties"]["longitude"])
        validate_coordinates(lat, lon)
        bbox = _geometry_bounds(feature["geometry"])
    else:
        # FIRMS country API applies Turkey's official boundary. These bounds
        # are descriptive in the response and are not sent to its area API.
        lat, lon = 39.0, 35.5
        bbox = (25.0, 35.0, 46.0, 43.0)

    province_name = (
        str(feature["properties"].get("province_name", feature["properties"].get("name", "")))
        if feature is not None else "Türkiye"
    )
    now_local = datetime.now(timezone(timedelta(hours=3))).isoformat()
    today_context = _harvest_context(
        datetime.now(timezone(timedelta(hours=3))).date(), province_name,
    ) if scope != "country" else None
    if today_context and today_context.get("region"):
        window_label = f"{today_context['window_start'][5:]}–{today_context['window_end'][5:]}"
        current_period = "şu an pencere içinde" if today_context.get("in_period") else "şu an pencere dışında"
        harvest_status = (
            f"{today_context['region']} hububat takvimi: yaklaşık {window_label} · {current_period}. "
            "Bu bölge ve hasat ilerlemesi eğitim modeli girdisidir."
        )
    else:
        harvest_status = "Hasat takvimi il/bölgeye göre değişir; nokta ayrıntısında yerel dönem gösterilir."
    result: dict[str, Any] = {
        "scope": scope,
        "district": district,
        "province_code": province,
        "province": province_name,
        "updated_at": now_local,
        "center": {"latitude": lat, "longitude": lon},
        "bbox": list(bbox),
        "boundary_source": "OpenStreetMap-derived · geoBoundaries ADM2" if scope == "district" else "OpenStreetMap-derived · geoBoundaries ADM1",
        "boundary_osm_id": feature["properties"].get("shapeID") if feature is not None else None,
        "hours": hours,
        "selected_day": day,
        "weather": None,
        "weather_error": None,
        "risk": None,
        "hotspots": [],
        "map_hotspots": [],
        "map_hotspots_truncated": False,
        "hotspots_total": 0,
        "hotspots_truncated": False,
        "hotspots_offset": offset,
        "hotspots_page_size": 40 if scope == "district" else 100,
        "hotspots_has_more": False,
        "hotspots_by_day": [],
        "firms_status": "NASA FIRMS durumu henüz sorgulanmadı.",
        "history_status": "Tarihsel kümelenme henüz hesaplanmadı.",
        "model_status": (
            "Model sıralaması için il ve ilçe seçin; Türkiye geneli tek hava noktasıyla sıralanmaz."
            if scope != "district" else "Seçili ilçe için canlı model girdileri bekleniyor."
        ),
        "model_validation": None,
        "harvest_calendar_status": harvest_status,
        "harvest_calendar": regional_calendar_payload(),
        "harvest_calendar_sources": HARVEST_SOURCES,
    }
    weather_frame: pd.DataFrame | None = None
    current: pd.Series | None = None
    def fetch_firms_for_scope() -> tuple[pd.DataFrame, str]:
        firms_key = secret_or_environment("FIRMS_MAP_KEY")
        if scope == "district":
            return fetch_nasa_firms_hotspots(bbox, firms_key, allow_demo=False)
        days = min(5, max(1, (hours + 23) // 24 + (1 if hours >= 24 else 0)))
        return fetch_nasa_firms_extent_hotspots(_turkey_bounds(), firms_key, days=days)

    # Weather and NASA downloads are independent network calls; start them
    # together so their latencies overlap on district views.
    data_pool = ThreadPoolExecutor(max_workers=2)
    firms_future = data_pool.submit(fetch_firms_for_scope)
    weather_future = (
        data_pool.submit(fetch_open_meteo_weather, round(lat, 4), round(lon, 4))
        if scope == "district" else None
    )
    if scope == "district":
        try:
            weather_frame, current, weather_source = weather_future.result()
            rain24 = previous_24h_rain(weather_frame, pd.Timestamp(current["time"]))
            indices = calculate_cffdrs_system(
                temp=float(current["temperature_2m"]),
                rh=float(current["relative_humidity_2m"]),
                ws=float(current["wind_speed_10m"]),
                prec=rain24,
                lat=lat,
                month=pd.Timestamp(current["time"]).month,
            )
            risk_label, risk_color, _ = classify_ffmc(indices["ffmc"])
            result["weather"] = {
                "source": weather_source,
                "time": _json_value(current["time"]),
                "temperature": float(current["temperature_2m"]),
                "humidity": float(current["relative_humidity_2m"]),
                "wind_speed": float(current["wind_speed_10m"]),
                "rain_24h": rain24,
                "forecast": [
                    {
                        "time": _json_value(row.time),
                        "temperature": _json_value(row.temperature_2m),
                        "humidity": _json_value(row.relative_humidity_2m),
                        "rain": _json_value(row.precipitation),
                    }
                    for row in weather_frame.loc[
                        weather_frame["time"].between(
                            pd.Timestamp(current["time"]),
                            pd.Timestamp(current["time"]) + pd.Timedelta(hours=24),
                        )
                    ].head(25).itertuples(index=False)
                ],
            }
            result["risk"] = {
                "ffmc": indices["ffmc"], "fwi": indices["fwi"],
                "label": risk_label, "color": risk_color,
            }
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            result["weather_error"] = str(exc)
    else:
        result["weather_error"] = "Hava özeti yalnız seçili ilçe için hesaplanır; ülke/il düzeyinde tek nokta yanıltıcı olur."

    try:
        # The country feed uses FIRMS' supported area endpoint over the actual
        # province-boundary extent; results are clipped to Turkey below.
        events, status = firms_future.result()
        if not events.empty:
            valid_coords = events["latitude"].notna() & events["longitude"].notna()
            events = events.loc[valid_coords].copy()
            if scope == "country":
                within_boundary = events.apply(
                    lambda row: _province_name_at(
                        float(row["longitude"]), float(row["latitude"]),
                    ) is not None,
                    axis=1,
                )
                events = events.loc[within_boundary].reset_index(drop=True)
            elif feature is not None:
                within_boundary = events.apply(
                    lambda row: _point_in_geometry(
                        float(row["longitude"]), float(row["latitude"]), feature["geometry"], bbox,
                    ),
                    axis=1,
                )
                events = events.loc[within_boundary].reset_index(drop=True)
        history_scores: dict[tuple[float, float], dict[str, Any]] = {}
        try:
            history_path = ROOT / "data" / "firms_cell_history.csv"
            snapshot_day = datetime.now(timezone.utc).date()
            successful_complete_snapshot = status.startswith("NASA FIRMS") and "geçici" not in status.casefold()
            todays_events = events.loc[
                pd.to_datetime(events["acq_date"], errors="coerce").dt.date == snapshot_day
            ].copy() if not events.empty else events.copy()
            history = read_history(history_path)
            if successful_complete_snapshot:
                history = record_snapshot(bbox, todays_events, history_path, snapshot_day)
                if not todays_events.empty:
                    scored_today, _, prior_days = score_snapshot(bbox, todays_events, history, snapshot_day)
                    for _, scored_row in scored_today.iterrows():
                        lat_value = pd.to_numeric(scored_row.get("latitude"), errors="coerce")
                        lon_value = pd.to_numeric(scored_row.get("longitude"), errors="coerce")
                        if pd.notna(lat_value) and pd.notna(lon_value):
                            history_scores[(round(float(lat_value), 5), round(float(lon_value), 5))] = {
                                "score": _json_value(scored_row.get("cluster_anomaly_score")),
                                "count": _json_value(scored_row.get("cluster_daily_count")),
                                "flag": bool(scored_row.get("cluster_anomaly", False)),
                            }
                else:
                    prior_days = int(history.loc[
                        (history.area_id == _area_id(bbox))
                        & (pd.to_datetime(history.date, errors="coerce").dt.date < snapshot_day)
                    ].date.nunique()) if not history.empty else 0
                if prior_days >= MIN_HISTORY_DAYS:
                    result["history_status"] = f"Tarihsel karşılaştırma açık · {prior_days} önceki günlük kapsam kaydı · {MIN_HISTORY_DAYS} gün gerekli"
                else:
                    result["history_status"] = f"Başlangıç dönemi · {prior_days}/{MIN_HISTORY_DAYS} önceki günlük kayıt. Eşik dolana kadar anomali puanı gösterilmez."
            else:
                result["history_status"] = "Günlük geçmiş güncellenmedi: FIRMS yanıtı eksik veya başarısız; eksik kaynak sıfır tespit sayılmadı."
        except Exception as exc:
            logging.warning("FIRMS geçmişi yazılamadı: %s", type(exc).__name__)
            result["history_status"] = "Tarihsel karşılaştırma kullanılamıyor; uydu bildirimleri gösterilmeye devam ediyor."
        recent = _recent_events(events, hours=hours)
        all_recent = recent.copy()
        if not all_recent.empty:
            daily_counts = all_recent.groupby("acq_date", dropna=True).size().sort_index(ascending=False)
            result["hotspots_by_day"] = [
                {"date": str(day_value), "count": int(count)} for day_value, count in daily_counts.items()
            ]
        if day and not recent.empty:
            recent = recent.loc[recent["acq_date"].astype(str) == day].copy().reset_index(drop=True)
        if not recent.empty:
            matching_count = len(recent)
            result["hotspots_total"] = int(matching_count)
            recent = recent.copy()
            hhmm = pd.to_numeric(recent["acq_time"], errors="coerce").map(
                lambda value: f"{int(value):04d}" if pd.notna(value) else ""
            )
            recent["_detection_time"] = pd.to_datetime(
                recent["acq_date"].astype(str) + " " + hhmm,
                format="%Y-%m-%d %H%M", errors="coerce", utc=True,
            )
            # Keep the alert list paginated, but send every current detection
            # to the country map. Previously only the newest 100 rows reached
            # Leaflet, making most provinces look empty until individually
            # opened. These lightweight map records retain raw observation
            # fields and deliberately do not invent CORINE context.
            map_limit = 5000
            map_recent = recent.sort_values(
                "_detection_time", ascending=False, na_position="last", kind="mergesort",
            ).head(map_limit)
            map_rows = []
            for _, map_row in map_recent.iterrows():
                lat_value = pd.to_numeric(map_row.get("latitude"), errors="coerce")
                lon_value = pd.to_numeric(map_row.get("longitude"), errors="coerce")
                if pd.isna(lat_value) or pd.isna(lon_value):
                    continue
                priority_score, priority_label, priority_color = anomaly_priority(map_row)
                candidate, _ = classify_hotspot_tr(map_row, None)
                map_province = _province_name_at(float(lon_value), float(lat_value)) if scope == "country" else province_name
                harvest_context = _harvest_context(
                    map_row.get("acq_date"), map_province,
                )
                point_history = history_scores.get(
                    (round(float(lat_value), 5), round(float(lon_value), 5)),
                    {},
                )
                raw_observation = {
                    key: _json_value(map_row.get(key))
                    for key in ("acq_date", "acq_time", "satellite", "confidence", "frp", "bright_ti4", "daynight")
                }
                map_rows.append({
                    "latitude": float(lat_value),
                    "longitude": float(lon_value),
                    **raw_observation,
                    "province": map_province,
                    "district": district,
                    "candidate": candidate,
                    "priority_score": priority_score if priority_score > 0 else None,
                    "priority_label": priority_label,
                    "priority_color": priority_color,
                    "harvest_context": harvest_context,
                    "cluster_anomaly_score": point_history.get("score"),
                    "cluster_daily_count": point_history.get("count"),
                    "cluster_anomaly": point_history.get("flag", False),
                    "observations": [raw_observation],
                    "latest_observation": raw_observation,
                    "map_only": True,
                })
            model_rank_by_event: dict[tuple[float, float, str, str, str], float] = {}
            if scope == "district" and map_rows:
                map_frame = pd.DataFrame(map_rows)
                map_frame, map_corine_status = enrich_hotspots_with_local_raster(map_frame)
                if not map_frame.get("corine_code", pd.Series(dtype="object")).notna().any():
                    # If the optional rasterio dependency or local raster is
                    # unavailable, retain the existing bounded Copernicus API
                    # lookup. Only matched points can receive a model rank.
                    map_frame, map_corine_status = enrich_hotspots_with_corine(map_frame)
                result["corine_status"] = map_corine_status
                try:
                    scored_map, model_status, model_validation = score_live_events(
                        events, map_frame, weather_frame,
                    )
                    result["model_status"] = model_status
                    result["model_validation"] = model_validation
                    for index, scored_row in scored_map.iterrows():
                        identity = _event_identity(scored_row)
                        score = pd.to_numeric(scored_row.get("model_rank_percentile"), errors="coerce")
                        if identity is not None and pd.notna(score):
                            model_rank_by_event[identity] = float(score)
                        if 0 <= int(index) < len(map_rows):
                            map_rows[int(index)]["corine_code"] = _json_value(scored_row.get("corine_code"))
                            map_rows[int(index)]["corine_class"] = _json_value(scored_row.get("corine_class"))
                            map_rows[int(index)]["model_rank_percentile"] = (
                                float(score) if pd.notna(score) else None
                            )
                except Exception as exc:
                    logging.warning("Canlı model sıralaması üretilemedi: %s", type(exc).__name__)
                    result["model_status"] = f"Model sıralaması hesaplanamadı ({type(exc).__name__})."
            elif scope != "district":
                result["model_status"] = "Model sıralaması için il ve ilçe seçin; ülke geneli tek hava noktasıyla puanlanmaz."
            result["map_hotspots"] = map_rows
            result["map_hotspots_truncated"] = matching_count > map_limit
            display_limit = 40 if scope == "district" else 100
            recent["_model_rank_sort"] = recent.apply(
                lambda row: model_rank_by_event.get(_event_identity(row), np.nan), axis=1,
            )
            recent = recent.sort_values(
                ["_model_rank_sort", "_detection_time"],
                ascending=[False, False], na_position="last", kind="mergesort",
            ).drop(columns=["_detection_time", "_model_rank_sort"])
            result["hotspots_page_size"] = display_limit
            recent = recent.iloc[offset:offset + display_limit].reset_index(drop=True)
            result["hotspots_offset"] = offset
            result["hotspots_has_more"] = offset + len(recent) < matching_count
            result["hotspots_truncated"] = result["hotspots_has_more"]
            recent, corine_status = enrich_hotspots_with_corine(recent)
            if not result.get("corine_status"):
                result["corine_status"] = corine_status
            rows = []
            for _, row in recent.iterrows():
                risk_context = result.get("risk") or {}
                candidate, _ = classify_hotspot_tr(row, risk_context.get("ffmc"))
                observations = _nearby_observations(events, row)
                latest_observation = observations[-1] if observations else None
                # The visible priority is recalculated from the latest actual
                # satellite observation available at refresh time. Missing data
                # stays missing; we never infer zero FRP from no repeat detection.
                score_row = row
                if latest_observation:
                    score_row = pd.Series({
                        "confidence": latest_observation.get("confidence"),
                        "frp": latest_observation.get("frp"),
                        "bright_ti4": latest_observation.get("bright_ti4"),
                    })
                priority_score, priority_label, priority_color = anomaly_priority(score_row)
                point_history = history_scores.get((round(float(row["latitude"]), 5), round(float(row["longitude"]), 5)), {})
                event_harvest = _harvest_context(row.get("acq_date"), province_name if scope != "country" else _province_name_at(
                    float(row["longitude"]), float(row["latitude"]),
                ))
                model_rank = model_rank_by_event.get(_event_identity(row))
                rows.append({
                    key: _json_value(row.get(key))
                    for key in (
                        "latitude", "longitude", "acq_date", "acq_time", "satellite",
                        "confidence", "frp", "bright_ti4", "daynight", "corine_code", "corine_class",
                    )
                } | {
                    "province": province_name if scope != "country" else _province_name_at(
                        float(row["longitude"]), float(row["latitude"]),
                    ),
                    "district": district,
                    "candidate": candidate,
                    "priority_score": priority_score if priority_score > 0 else None,
                    "priority_label": priority_label,
                    "priority_color": priority_color,
                    "cluster_anomaly_score": point_history.get("score"),
                    "cluster_daily_count": point_history.get("count"),
                    "cluster_anomaly": point_history.get("flag", False),
                    "model_rank_percentile": model_rank,
                    "harvest_context": event_harvest,
                    "observations": observations,
                    "latest_observation": latest_observation,
                    "data_checked_at_utc": datetime.now(timezone.utc).isoformat(),
                })
            result["hotspots"] = rows
        elif scope == "district":
            result["model_status"] = "Seçili zaman aralığında FIRMS kaydı yok; Model sıralaması üretilemedi."
        result["firms_status"] = status
    except Exception as exc:
        logging.exception("Dashboard FIRMS kaydı hazırlanamadı")
        result["firms_status"] = f"NASA FIRMS verisi alınamadı: {type(exc).__name__}"
    finally:
        data_pool.shutdown(wait=True, cancel_futures=True)
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.web_api:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
