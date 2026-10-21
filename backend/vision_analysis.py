"""Esri overhead imagery and optional visual land-cover classification.

The model describes visible land-cover context only. A basemap image cannot prove
that a FIRMS detection is a fire, stubble burning, or an illegal activity.
"""

from __future__ import annotations

import io
import json
import math
from urllib.parse import urlencode

import requests
import streamlit as st
from PIL import Image

ESRI_EXPORT_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
)
VISION_MODEL_ID = "gemini-3.8-flash"
VISION_GENERATE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{VISION_MODEL_ID}:generateContent"
)
LAND_COVER_LABELS = {
    "agriculture": "Tarım alanı",
    "industrial": "Sanayi / tesis alanı",
    "forest": "Ormanlık alan",
    "other": "Diğer arazi kullanımı",
    "uncertain": "Görüntüden ayırt edilemiyor",
}
SURFACE_STATES = (
    "standing_crop", "harvested_stubble", "baled_straw", "tilled_bare_soil",
    "dark_burn_scar", "visible_smoke_or_flame", "green_vegetation", "built_up",
    "forest", "water", "uncertain",
)
SURFACE_STATE_LABELS = {
    "standing_crop": "Ayaktaki ürün", "harvested_stubble": "Hasat edilmiş tarla görünümü",
    "baled_straw": "Balyalanmış saman görünümü", "tilled_bare_soil": "Sürülmüş çıplak toprak",
    "dark_burn_scar": "Koyu yüzey izi (olayla ilişkilendirilemez)",
    "visible_smoke_or_flame": "Duman/alev görünümü (olay teyidi değildir)",
    "green_vegetation": "Yeşil bitki örtüsü", "built_up": "Yapılaşmış alan",
    "forest": "Orman örtüsü", "water": "Su yüzeyi", "uncertain": "Görüntüden ayırt edilemiyor",
}


def _validate_coordinates(latitude: float, longitude: float) -> tuple[float, float]:
    lat, lon = float(latitude), float(longitude)
    if not math.isfinite(lat) or not math.isfinite(lon):
        raise ValueError("Uydu görüntüsü için koordinatlar sonlu sayı olmalı.")
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("Enlem veya boylam aralık dışında.")
    return lat, lon


def esri_export_url(latitude: float, longitude: float, size: int = 400) -> str:
    """Build an Esri World Imagery export URL centered on one FIRMS point."""
    lat, lon = _validate_coordinates(latitude, longitude)
    if not 128 <= int(size) <= 1024:
        raise ValueError("Görüntü boyutu 128–1024 piksel aralığında olmalı.")
    # A roughly 600 m square footprint. Esri imagery has no guaranteed capture
    # time and must never be described as a live image of the event.
    half_lat = 0.003
    half_lon = half_lat / max(math.cos(math.radians(lat)), 0.2)
    params = {
        "bbox": f"{lon-half_lon:.7f},{lat-half_lat:.7f},{lon+half_lon:.7f},{lat+half_lat:.7f}",
        "bboxSR": "4326",
        "imageSR": "4326",
        "size": f"{int(size)},{int(size)}",
        "format": "jpg",
        "f": "image",
    }
    return f"{ESRI_EXPORT_URL}?{urlencode(params)}"


@st.cache_data(ttl=60 * 60 * 24, max_entries=256, show_spinner=False)
def fetch_esri_snapshot(latitude: float, longitude: float, size: int = 400) -> bytes:
    """Fetch and normalize a 400×400 Esri World Imagery overview image."""
    url = esri_export_url(latitude, longitude, size)
    response = requests.get(
        url,
        headers={"User-Agent": "MaydonOs-FIRMS-Prototype/1.0"},
        timeout=(5, 25),
    )
    response.raise_for_status()
    if not response.headers.get("content-type", "").lower().startswith("image/"):
        raise ValueError("Esri yanıtı görsel değil; uydu görüntüsü alınamadı.")
    if not response.content:
        raise ValueError("Esri boş uydu görüntüsü döndürdü.")
    try:
        with Image.open(io.BytesIO(response.content)) as image:
            image = image.convert("RGB").resize((int(size), int(size)))
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=88, optimize=True)
            return output.getvalue()
    except (OSError, ValueError) as exc:
        raise ValueError("Esri görüntü dosyası çözümlenemedi.") from exc


@st.cache_data(ttl=60 * 60 * 24 * 30, max_entries=512, show_spinner=False)
def analyze_overhead_scene(api_key: str, image_bytes: bytes) -> dict[str, str]:
    """Classify visible land-cover context without asserting event cause."""
    if not api_key or not api_key.strip():
        raise ValueError("Görsel sınıflandırma API anahtarı girilmedi.")
    if not image_bytes:
        raise ValueError("Analiz için uydu görüntüsü boş.")
    import base64

    prompt = (
        "Review this small overhead basemap image centered on a NASA FIRMS coordinate. First describe only "
        "directly visible visual evidence (field boundaries, rows, buildings, vegetation and surface colors); "
        "then assign a broad land-cover class and a surface-state label for the image itself. The image date is "
        "unknown and may differ from the FIRMS acquisition date. It is not an event-time image. Never claim a "
        "fire, crop-residue burning, illegal activity, smoke event or cause is confirmed. Do not infer a crop "
        "species. Because image timing is unknown, use uncertain for dark_burn_scar and "
        "visible_smoke_or_flame; dark soil or shadows are not proof of burning. If details are unclear, use "
        "uncertain. Return JSON only and include brief limitations."
    )
    schema = {
        "type": "OBJECT",
        "properties": {
            "visual_evidence": {"type": "STRING"},
            "surface_state": {"type": "STRING", "enum": list(SURFACE_STATES)},
            "land_cover": {"type": "STRING", "enum": list(LAND_COVER_LABELS)},
            "description": {"type": "STRING"},
            "confidence": {"type": "STRING", "enum": ["low", "medium", "high"]},
            "limitations": {"type": "STRING"},
        },
        "required": ["visual_evidence", "surface_state", "land_cover", "description", "confidence", "limitations"],
        "propertyOrdering": ["visual_evidence", "surface_state", "land_cover", "description", "confidence", "limitations"],
    }
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_bytes).decode("ascii"),
                }},
            ]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }
    response = requests.post(
        VISION_GENERATE_URL,
        headers={"x-goog-api-key": api_key.strip(), "Content-Type": "application/json"},
        json=payload,
        timeout=(10, 60),
    )
    if response.status_code >= 400:
        try:
            details = response.json().get("error", {}).get("message", "")
        except (ValueError, AttributeError):
            details = ""
        raise RuntimeError(f"Görsel sınıflandırma API HTTP {response.status_code}: {details or response.reason}")
    try:
        response.raise_for_status()
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(text)
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Görsel sınıflandırma servisi geçerli yapılandırılmış yanıt döndürmedi.") from exc
    if result.get("land_cover") not in LAND_COVER_LABELS:
        raise RuntimeError("Yanıtta geçerli arazi sınıfı bulunamadı.")
    if result.get("surface_state") not in SURFACE_STATES:
        raise RuntimeError("Yanıtta geçerli yüzey görünümü bulunamadı.")
    if any(key not in result for key in schema["required"]):
        raise RuntimeError("Yanıtta zorunlu görsel analiz alanları eksik.")
    return {key: str(result[key]) for key in schema["required"]}

