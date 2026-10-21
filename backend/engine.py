"""Data and calculation functions for the MaydonOs fire-weather prototype."""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    import openmeteo_requests
except ImportError:  # Keep the app renderable so it can explain setup errors.
    openmeteo_requests = None

try:
    import requests_cache
    from retry_requests import retry
except ImportError:  # Report a clear setup message if reliability extras are missing.
    requests_cache = None
    retry = None

try:
    from cffdrs.fwi import (
        buildup_index,
        drought_code,
        duff_moisture_code,
        fire_weather_index,
        fine_fuel_moisture_code,
        initial_spread_index,
    )
    from cffdrs.hourly_fine_fuel_moisture_code import hourly_fine_fuel_moisture_code
except ImportError:
    buildup_index = None
    drought_code = None
    duff_moisture_code = None
    fire_weather_index = None
    fine_fuel_moisture_code = None
    initial_spread_index = None
    hourly_fine_fuel_moisture_code = None

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{bbox}/{days}"
FIRMS_SOURCES = (
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
    "MODIS_NRT",
)
REQUEST_TIMEOUT = (6, 15)
FIRMS_TIMEOUT = (4, 12)
# Five calendar days are needed to cover an exact rolling 96-hour UI window
# when the request is made late in the day (FIRMS area API maximum is 5 days).
FIRMS_LOOKBACK_DAYS = 5
WEATHER_COLUMNS = ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation")
WEATHER_CONTEXT_COLUMNS = (
    "wind_direction_10m", "vapour_pressure_deficit", "soil_moisture_0_to_7cm",
)
FIRMS_COLUMNS = (
    "latitude", "longitude", "bright_ti4", "scan", "track", "acq_date", "acq_time",
    "satellite", "confidence", "version", "bright_ti5", "frp", "daynight", "firms_source",
)


def validate_coordinates(lat: float, lon: float) -> None:
    if not (math.isfinite(float(lat)) and math.isfinite(float(lon))):
        raise ValueError("Koordinatlar geçerli, sonlu sayılar olmalı.")
    if not (35.0 <= float(lat) <= 43.0 and 25.0 <= float(lon) <= 46.0):
        raise ValueError("Koordinat Türkiye için belirlenen çalışma alanının dışında.")


def bbox_from_center(lat: float, lon: float, radius: float) -> tuple[float, float, float, float]:
    validate_coordinates(lat, lon)
    if not math.isfinite(float(radius)) or not 0.02 <= float(radius) <= 0.75:
        raise ValueError("Harita yarıçapı 0,02° ile 0,75° arasında olmalı.")
    return float(lon) - radius, float(lat) - radius, float(lon) + radius, float(lat) + radius


def validate_weather(df: pd.DataFrame) -> pd.DataFrame:
    required = {"time", *WEATHER_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Hava yanıtında eksik alan var: {', '.join(sorted(missing))}.")
    optional_columns = [name for name in WEATHER_CONTEXT_COLUMNS if name in df.columns]
    result = df.loc[:, ["time", *WEATHER_COLUMNS, *optional_columns]].copy()
    result["time"] = pd.to_datetime(result["time"], errors="coerce")
    if result["time"].isna().any():
        raise ValueError("Hava verisinde geçersiz zaman bilgisi var.")
    for col in WEATHER_COLUMNS:
        result[col] = pd.to_numeric(result[col], errors="coerce")
        if not np.isfinite(result[col].to_numpy(dtype=float)).all():
            raise ValueError(f"{col} alanında eksik veya geçersiz veri var; değer doldurulmadı.")
    for col in optional_columns:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    if not result["relative_humidity_2m"].between(0, 100).all():
        raise ValueError("Bağıl nem %0–100 aralığında olmalı.")
    if (result["wind_speed_10m"] < 0).any() or (result["precipitation"] < 0).any():
        raise ValueError("Rüzgâr ve yağış negatif olamaz.")
    if not result["temperature_2m"].between(-60, 60).all():
        raise ValueError("Sıcaklık fiziksel kontrol sınırlarının dışında.")
    result = result.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    if result.empty:
        raise ValueError("Open-Meteo boş hava verisi döndürdü.")
    return result


@st.cache_resource(show_spinner=False)
def _openmeteo_client():
    """Build a cached Open-Meteo client with bounded retries and request timeouts."""
    if openmeteo_requests is None:
        raise RuntimeError("openmeteo-requests kurulu değil. Proje klasöründe pip install -r requirements.txt çalıştırın.")
    if requests_cache is None or retry is None:
        raise RuntimeError("Open-Meteo için requests-cache ve retry-requests eksik. requirements.txt dosyasını yeniden kurun.")
    class TimeoutCachedSession(requests_cache.CachedSession):
        def request(self, method, url, **kwargs):
            kwargs.setdefault("timeout", REQUEST_TIMEOUT)
            return super().request(method, url, **kwargs)

    cache_session = TimeoutCachedSession(
        cache_name=".openmeteo_cache",
        backend="memory",
        expire_after=900,
    )
    retry_session = retry(cache_session, retries=1, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_open_meteo_weather(lat: float, lon: float) -> tuple[pd.DataFrame, pd.Series, str]:
    """Fetch Open-Meteo current conditions plus a 48h history/24h forecast window."""
    validate_coordinates(lat, lon)
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "current": list(WEATHER_COLUMNS),
        "hourly": [*WEATHER_COLUMNS, *WEATHER_CONTEXT_COLUMNS],
        "past_days": 2,
        "forecast_days": 1,
        "timezone": "Europe/Istanbul",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    try:
        client = _openmeteo_client()
        # openmeteo-requests forwards HTTP options through its supplied session;
        # TimeoutCachedSession bounds the request and keeps retries limited.
        responses = client.weather_api(OPEN_METEO_URL, params=params)
        if not responses:
            raise ValueError("Open-Meteo boş yanıt döndürdü.")
        response = responses[0]
        hourly = response.Hourly()
        all_hourly = [*WEATHER_COLUMNS, *WEATHER_CONTEXT_COLUMNS]
        if hourly is None or hourly.VariablesLength() < len(all_hourly):
            raise ValueError("Open-Meteo saatlik değişkenleri eksik döndürdü.")
        offset = int(response.UtcOffsetSeconds())
        epochs = np.arange(hourly.Time(), hourly.TimeEnd(), hourly.Interval(), dtype=np.int64)
        local_times = pd.to_datetime(epochs + offset, unit="s")
        values = {
            name: hourly.Variables(i).ValuesAsNumpy()
            for i, name in enumerate(all_hourly)
        }
        frame = validate_weather(pd.DataFrame({"time": local_times, **values}))

        current_block = response.Current()
        current_time = pd.to_datetime(int(current_block.Time()) + offset, unit="s")
        current_values = {
            name: float(current_block.Variables(i).Value())
            for i, name in enumerate(WEATHER_COLUMNS)
        }
        current = pd.Series({"time": current_time, **current_values})
        # Validate the separate current block using the same rules as hourly data.
        validate_weather(pd.DataFrame([current.to_dict()]))
        return frame, current, "Open-Meteo • saatlik hava verisi"
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        message = str(exc).replace("\n", " ")[:240]
        raise RuntimeError(f"Hava verisi alınamadı veya doğrulanamadı: {message}") from exc


# Keep the older engine name available for existing scripts.
get_weather_data = fetch_open_meteo_weather


def calculate_fire_risk(
    temp: float,
    rh: float,
    ws: float,
    prec: float,
    prev_ffmc: float = 85.0,
) -> float:
    """Calculate one CFFDRS FFMC step with the official `cffdrs` implementation.

    CFFDRS is a daily fire-weather model. The app feeds a current weather snapshot
    and the preceding 24h precipitation as a provisional nowcast, not as an
    independently validated local fire-probability model.
    """
    vals = tuple(float(x) for x in (temp, rh, ws, prec, prev_ffmc))
    if not all(math.isfinite(x) for x in vals):
        raise ValueError("FFMC girdileri eksik veya geçersiz.")
    t, h, wind, rain, previous = vals
    if not -60 <= t <= 60 or not 0 <= h <= 100 or wind < 0 or rain < 0 or not 0 <= previous <= 101:
        raise ValueError("FFMC girdileri fiziksel sınırların dışında.")
    if fine_fuel_moisture_code is None:
        raise RuntimeError("cffdrs modülü kurulu değil. Proje klasöründe pip install -r requirements.txt çalıştırın.")
    result = fine_fuel_moisture_code(
        ffmc_yda=previous, temp=t, rh=h, ws=wind, prec=rain
    )
    score = float(result)
    if not math.isfinite(score):
        raise ValueError("CFFDRS geçerli bir FFMC değeri üretmedi.")
    return min(99.0, max(0.0, score))


def calculate_cffdrs_system(
    temp: float,
    rh: float,
    ws: float,
    prec: float,
    lat: float,
    month: int,
    prev_ffmc: float = 85.0,
    prev_dmc: float = 6.0,
    prev_dc: float = 15.0,
) -> dict[str, float]:
    """Calculate the complete daily FWI component chain from cffdrs_py.

    Previous-day codes are explicit inputs. Defaults are only starting values,
    not locally observed or calibrated codes; the UI discloses this clearly.
    """
    inputs = tuple(float(v) for v in (temp, rh, ws, prec, lat, prev_ffmc, prev_dmc, prev_dc))
    if not all(math.isfinite(v) for v in inputs):
        raise ValueError("CFFDRS girdilerinde eksik veya geçersiz değer var.")
    t, humidity, wind, rain, latitude, ffmc_yesterday, dmc_yesterday, dc_yesterday = inputs
    if not -60 <= t <= 60 or not 0 <= humidity <= 100 or wind < 0 or rain < 0:
        raise ValueError("Hava değerleri CFFDRS fiziksel sınırlarının dışında.")
    if not -90 <= latitude <= 90 or int(month) != month or not 1 <= int(month) <= 12:
        raise ValueError("CFFDRS için enlem veya ay değeri geçersiz.")
    if not 0 <= ffmc_yesterday <= 101 or dmc_yesterday < 0 or dc_yesterday < 0:
        raise ValueError("Önceki FFMC, DMC veya DC kodu geçersiz.")
    required = (fine_fuel_moisture_code, duff_moisture_code, drought_code,
                initial_spread_index, buildup_index, fire_weather_index)
    if any(function is None for function in required):
        raise RuntimeError("CFFDRS bileşenleri kurulu değil. requirements.txt dosyasını yeniden kurun.")

    ffmc = float(fine_fuel_moisture_code(
        ffmc_yda=ffmc_yesterday, temp=t, rh=humidity, ws=wind, prec=rain
    ))
    dmc = float(duff_moisture_code(
        dmc_yda=dmc_yesterday, temp=t, rh=humidity, prec=rain,
        lat=latitude, mon=int(month), lat_adjust=True,
    ))
    dc = float(drought_code(
        dc_yda=dc_yesterday, temp=t, rh=humidity, prec=rain,
        lat=latitude, mon=int(month), lat_adjust=True,
    ))
    isi = float(initial_spread_index(ffmc=ffmc, ws=wind))
    bui = float(buildup_index(dmc=dmc, dc=dc))
    fwi = float(fire_weather_index(isi=isi, bui=bui))
    results = {"ffmc": ffmc, "dmc": dmc, "dc": dc, "isi": isi, "bui": bui, "fwi": fwi}
    if not all(math.isfinite(value) for value in results.values()):
        raise ValueError("CFFDRS geçerli olmayan bir bileşen değeri üretti.")
    return results


def calculate_hourly_ffmc_series(
    weather: pd.DataFrame,
    as_of: pd.Timestamp,
    initial_ffmc: float = 85.0,
) -> pd.DataFrame:
    """Run CFFDRS hourly FFMC sequentially over complete hourly weather data."""
    if hourly_fine_fuel_moisture_code is None:
        raise RuntimeError("CFFDRS saatlik FFMC bileşeni kurulu değil.")
    if not math.isfinite(float(initial_ffmc)) or not 0 <= float(initial_ffmc) <= 101:
        raise ValueError("Başlangıç FFMC kodu 0–101 arasında olmalı.")
    frame = validate_weather(weather)
    end = pd.Timestamp(as_of)
    frame = frame.loc[frame["time"] <= end].copy().reset_index(drop=True)
    if len(frame) < 25:
        raise ValueError("Saatlik FFMC için son 24 saati kapsayan 25 saatlik kayıt gerekli.")
    frame = frame.tail(25).reset_index(drop=True)
    gaps = frame["time"].diff().dropna()
    if not gaps.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Saatlik FFMC için hava serisinde kesintisiz saatlik kayıt bulunmalı.")

    values: list[float] = [float(initial_ffmc)]
    previous = float(initial_ffmc)
    for row in frame.iloc[1:].itertuples(index=False):
        value = float(hourly_fine_fuel_moisture_code(
            temp=float(row.temperature_2m),
            rh=float(row.relative_humidity_2m),
            ws=float(row.wind_speed_10m),
            prec=float(row.precipitation),
            fo=previous,
            t0=1,
        ))
        if not math.isfinite(value):
            raise ValueError("Saatlik FFMC geçerli olmayan bir değer üretti.")
        values.append(value)
        previous = value
    frame["ffmc_hourly"] = values
    return frame


def previous_24h_rain(weather: pd.DataFrame, as_of: pd.Timestamp) -> float:
    """Sum observed hourly precipitation for a complete preceding 24h window."""
    end = pd.Timestamp(as_of)
    window = weather.loc[(weather["time"] > end - pd.Timedelta(hours=24)) & (weather["time"] <= end)]
    if len(window) != 24 or not window["time"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("FFMC için önceki 24 saatin kesintisiz yağış verisi yok.")
    return float(window["precipitation"].sum())


def classify_ffmc(value: float) -> tuple[str, str, str]:
    """User-requested display bands; these are not Turkish official warning levels."""
    if value < 70:
        return "Düşük gösterge", "#26734d", "🟢"
    if value < 85:
        return "Orta gösterge", "#b8860b", "🟡"
    return "Yüksek gösterge", "#b9382f", "🔴"


def validate_bbox(bbox: tuple[float, float, float, float]) -> None:
    west, south, east, north = map(float, bbox)
    if not all(math.isfinite(v) for v in (west, south, east, north)):
        raise ValueError("Harita sınırı geçersiz koordinat içeriyor.")
    if not (25 <= west < east <= 46 and 35 <= south < north <= 43):
        raise ValueError("Harita alanı çalışma sınırlarının dışında veya hatalı.")
    if (east - west) * (north - south) > 1.5:
        raise ValueError("FIRMS sorgu alanı çok büyük; harita ölçeğini küçültün.")


def validate_firms(df: pd.DataFrame, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    validate_bbox(bbox)
    if df.empty:
        return pd.DataFrame(columns=[*FIRMS_COLUMNS, "is_demo"])
    required = {"latitude", "longitude", "acq_date", "acq_time", "confidence", "daynight"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"FIRMS yanıtında gerekli alan eksik: {', '.join(sorted(missing))}.")
    result = df.copy()
    for col in ("latitude", "longitude"):
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result = result.dropna(subset=["latitude", "longitude"])
    west, south, east, north = bbox
    result = result.loc[result["latitude"].between(south, north) & result["longitude"].between(west, east)].copy()
    for col in FIRMS_COLUMNS:
        if col not in result:
            result[col] = np.nan
    if "is_demo" not in result:
        result["is_demo"] = False
    return result.reset_index(drop=True)


def demo_hotspots(bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    """Generate deterministic, conspicuously synthetic points for a UI demo only."""
    validate_bbox(bbox)
    west, south, east, north = bbox
    digest = hashlib.sha256(",".join(f"{x:.4f}" for x in bbox).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:4], "big"))
    count = 4
    local_now = datetime.now(timezone(timedelta(hours=3)))
    latest_night = local_now.replace(hour=2, minute=20, second=0, microsecond=0)
    if latest_night > local_now:
        latest_night -= timedelta(days=1)
    if 9 <= local_now.hour < 18:
        latest_day = local_now - timedelta(minutes=30)
    elif local_now.hour >= 18:
        latest_day = local_now.replace(hour=17, minute=30, second=0, microsecond=0)
    else:
        latest_day = (local_now - timedelta(days=1)).replace(hour=17, minute=30, second=0, microsecond=0)
    local_detection_times = [
        latest_night.replace(hour=1, minute=40),
        latest_night.replace(hour=2, minute=20),
        latest_day,
        latest_day - timedelta(minutes=30),
    ]
    anchors = [(37.8478, 40.6628), (37.7253, 40.4156)]  # Bismil and Çınar centers
    west, south, east, north = bbox
    inside_anchors = [(a_lat, a_lon) for a_lat, a_lon in anchors if south <= a_lat <= north and west <= a_lon <= east]
    if not inside_anchors:
        inside_anchors = [((south + north) / 2, (west + east) / 2)]
    records = []
    for i in range(count):
        anchor_lat, anchor_lon = inside_anchors[i % len(inside_anchors)]
        stamp = local_detection_times[i].astimezone(timezone.utc)
        records.append({
            "latitude": float(np.clip(anchor_lat + rng.uniform(-0.02, 0.02), south, north)),
            "longitude": float(np.clip(anchor_lon + rng.uniform(-0.02, 0.02), west, east)),
            "bright_ti4": float(rng.uniform(315, 350)), "scan": 0.4, "track": 0.4,
            "acq_date": stamp.strftime("%Y-%m-%d"), "acq_time": stamp.strftime("%H%M"),
            "satellite": "DEMO", "confidence": "nominal", "version": "synthetic-demo",
            "bright_ti5": np.nan, "frp": float(rng.uniform(1, 12)), "daynight": "N" if i < 2 else "D",
            "is_demo": True,
        })
    return pd.DataFrame(records, columns=[*FIRMS_COLUMNS, "is_demo"])


def _firms_error(message: str, bbox: tuple[float, float, float, float], allow_demo: bool):
    if allow_demo:
        return demo_hotspots(bbox), f"SENTETİK DEMO • Gerçek NASA FIRMS verisi değil • {message}"
    return pd.DataFrame(columns=[*FIRMS_COLUMNS, "is_demo"]), message


class FirmsUnavailable(RuntimeError):
    """A live FIRMS query failed; exceptions must not become cached results."""


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_firms_success(
    bbox: tuple[float, float, float, float], map_key: str
) -> tuple[pd.DataFrame, str]:
    """Fetch recent detections from VIIRS S-NPP/NOAA-20/NOAA-21 and MODIS.

    Successful sensor feeds are returned even if another feed is temporarily
    unavailable. The status reports partial coverage instead of concealing it.
    """
    validate_bbox(bbox)
    if not map_key or not map_key.strip():
        raise FirmsUnavailable("FIRMS MAP_KEY girilmedi.")
    clean_key = map_key.strip()
    if re.fullmatch(r"[0-9a-fA-F]{32}", clean_key) is None:
        raise FirmsUnavailable("FIRMS MAP_KEY biçimi geçersiz.")
    west, south, east, north = bbox
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def fetch_source(source: str) -> tuple[str, pd.DataFrame | None, str | None]:
        url = FIRMS_AREA_URL.format(
            key=clean_key,
            source=source,
            bbox=f"{west:.4f},{south:.4f},{east:.4f},{north:.4f}",
            days=FIRMS_LOOKBACK_DAYS,
        )
        try:
            response = requests.get(url, timeout=FIRMS_TIMEOUT)
            response.raise_for_status()
            body = response.text.strip()
            if not body:
                return source, None, "boş yanıt"
            frame = pd.read_csv(io.StringIO(body))
            result = validate_firms(frame, bbox)
            result["is_demo"] = False
            result["firms_source"] = source
            return source, result, None
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            name = type(exc).__name__
            if name == "SSLError":
                message = "TLS sertifikası doğrulanamadı"
            elif name in {"ConnectTimeout", "Timeout", "ReadTimeout"}:
                message = "zaman aşımı"
            elif getattr(getattr(exc, "response", None), "status_code", None) in (401, 403):
                message = "MAP_KEY reddedildi"
            elif getattr(getattr(exc, "response", None), "status_code", None) == 429:
                message = "istek sınırı"
            else:
                message = "bağlantı hatası"
            return source, None, message

    frames: list[pd.DataFrame] = []
    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=len(FIRMS_SOURCES)) as pool:
        futures = [pool.submit(fetch_source, source) for source in FIRMS_SOURCES]
        for future in as_completed(futures):
            source, frame, error = future.result()
            if error is None and frame is not None:
                succeeded.append(source)
                frames.append(frame)
            else:
                failed.append((source, error or "yanıt okunamadı"))

    if not frames:
        reasons = {reason for _, reason in failed}
        if "TLS sertifikası doğrulanamadı" in reasons:
            message = "FIRMS TLS sertifikası doğrulanamadı. Ağdaki HTTPS denetimini kontrol edin."
        elif "MAP_KEY reddedildi" in reasons:
            message = "FIRMS isteği reddedildi; MAP_KEY değerini kontrol edin."
        elif "istek sınırı" in reasons:
            message = "FIRMS istek sınırına ulaşıldı; biraz sonra yeniden deneyin."
        elif "zaman aşımı" in reasons:
            message = "FIRMS bağlantısı zaman aşımına uğradı."
        else:
            message = "NASA FIRMS verisi alınamadı. İnternet ve servis erişimini kontrol edin."
        raise FirmsUnavailable(message)

    result = pd.concat(frames, ignore_index=True)
    # Remove only repeated sensor records with identical time and location;
    # close detections from different sensors remain independently traceable.
    result = result.drop_duplicates(
        subset=["latitude", "longitude", "acq_date", "acq_time", "satellite"], keep="first"
    ).reset_index(drop=True)
    succeeded.sort()
    source_labels = {
        "VIIRS_SNPP_NRT": "VIIRS S-NPP", "VIIRS_NOAA20_NRT": "VIIRS NOAA-20",
        "VIIRS_NOAA21_NRT": "VIIRS NOAA-21", "MODIS_NRT": "MODIS",
    }
    status = "NASA FIRMS • " + ", ".join(source_labels.get(source, source) for source in succeeded)
    if failed:
        status += f" • {len(failed)} sensör kaynağı geçici olarak alınamadı"
    return result, status


def _validate_firms_extent(
    frame: pd.DataFrame, bbox: tuple[float, float, float, float],
) -> pd.DataFrame:
    """Validate FIRMS rows for a bounded national query without the local-area cap."""
    west, south, east, north = map(float, bbox)
    if not all(math.isfinite(value) for value in (west, south, east, north)):
        raise ValueError("FIRMS ülke sınırı geçersiz koordinat içeriyor.")
    if not (24 <= west < east <= 47 and 34 <= south < north <= 43 and (east - west) * (north - south) <= 200):
        raise ValueError("FIRMS ulusal sorgu sınırı geçersiz veya fazla geniş.")
    required = {"latitude", "longitude", "acq_date", "acq_time", "confidence", "daynight"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"FIRMS yanıtında alan eksik: {', '.join(sorted(missing))}.")
    result = frame.copy()
    for column in ("latitude", "longitude"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["latitude", "longitude"])
    result = result.loc[
        result["latitude"].between(south, north)
        & result["longitude"].between(west, east)
    ].copy()
    for column in FIRMS_COLUMNS:
        if column not in result:
            result[column] = np.nan
    result["is_demo"] = False
    return result.reset_index(drop=True)


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_firms_extent_success(
    bbox: tuple[float, float, float, float], map_key: str, days: int,
) -> tuple[pd.DataFrame, str]:
    """Fetch real FIRMS records over a bounded national extent from the area API."""
    _validate_firms_extent(pd.DataFrame(columns=FIRMS_COLUMNS), bbox)
    if not map_key or re.fullmatch(r"[0-9a-fA-F]{32}", map_key.strip()) is None:
        raise FirmsUnavailable("FIRMS MAP_KEY girilmedi veya biçimi geçersiz.")
    if not 1 <= int(days) <= 5:
        raise FirmsUnavailable("FIRMS sorgusu 1–5 gün aralığında olmalı.")
    west, south, east, north = map(float, bbox)
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Broad national requests often time out at FIRMS. Query six smaller,
    # overlapping-free rectangles in parallel and merge the actual observations.
    lon_edges = np.linspace(west, east, 4)
    lat_edges = np.linspace(south, north, 3)
    tiles = [
        (float(lon_edges[ix]), float(lat_edges[iy]), float(lon_edges[ix + 1]), float(lat_edges[iy + 1]))
        for iy in range(len(lat_edges) - 1)
        for ix in range(len(lon_edges) - 1)
    ]

    def fetch_source(source: str, tile: tuple[float, float, float, float]) -> tuple[str, tuple[float, float, float, float], pd.DataFrame | None, str | None]:
        tile_west, tile_south, tile_east, tile_north = tile
        url = FIRMS_AREA_URL.format(
            key=map_key.strip(), source=source,
            bbox=f"{tile_west:.4f},{tile_south:.4f},{tile_east:.4f},{tile_north:.4f}",
            days=int(days),
        )
        try:
            response = requests.get(url, timeout=(6, 20))
            response.raise_for_status()
            body = response.text.strip()
            if not body:
                return source, tile, None, "boş yanıt"
            frame = pd.read_csv(io.StringIO(body))
            result = _validate_firms_extent(frame, tile)
            result["firms_source"] = source
            return source, tile, result, None
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if type(exc).__name__ == "SSLError":
                message = "TLS sertifikası doğrulanamadı"
            elif type(exc).__name__ in {"ConnectTimeout", "Timeout", "ReadTimeout"}:
                message = "zaman aşımı"
            elif status_code in (401, 403):
                message = "MAP_KEY reddedildi"
            elif status_code == 429:
                message = "istek sınırı"
            elif status_code == 400:
                message = "geçersiz FIRMS alan sorgusu (HTTP 400)"
            else:
                message = f"FIRMS yanıt hatası (HTTP {status_code})" if status_code else "bağlantı hatası"
            return source, tile, None, message

    frames: list[pd.DataFrame] = []
    tile_results: dict[str, set[tuple[float, float, float, float]]] = {source: set() for source in FIRMS_SOURCES}
    failed: list[tuple[str, tuple[float, float, float, float], str]] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch_source, source, tile) for source in FIRMS_SOURCES for tile in tiles]
        for future in as_completed(futures):
            source, tile, frame, error = future.result()
            if error is None and frame is not None:
                tile_results[source].add(tile)
                frames.append(frame)
            else:
                failed.append((source, tile, error or "yanıt okunamadı"))
    if not frames:
        reasons = {reason for _, _, reason in failed}
        if "TLS sertifikası doğrulanamadı" in reasons:
            message = "FIRMS TLS sertifikası doğrulanamadı. Ağdaki HTTPS denetimini kontrol edin."
        elif "MAP_KEY reddedildi" in reasons:
            message = "FIRMS isteği reddedildi; MAP_KEY değerini kontrol edin."
        elif "istek sınırı" in reasons:
            message = "FIRMS istek sınırına ulaşıldı; biraz sonra yeniden deneyin."
        elif "zaman aşımı" in reasons:
            message = "FIRMS Türkiye alan sorgusu zaman aşımına uğradı."
        elif any("HTTP 400" in reason for reason in reasons):
            message = "NASA FIRMS Türkiye alan sorgusunu kabul etmedi (HTTP 400)."
        else:
            detail = ", ".join(sorted(reasons)) or "boş yanıt"
            message = f"NASA FIRMS Türkiye verisi alınamadı: {detail}."
        raise FirmsUnavailable(message)

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(
        subset=["latitude", "longitude", "acq_date", "acq_time", "satellite"], keep="first"
    ).reset_index(drop=True)
    source_labels = {
        "VIIRS_SNPP_NRT": "VIIRS S-NPP", "VIIRS_NOAA20_NRT": "VIIRS NOAA-20",
        "VIIRS_NOAA21_NRT": "VIIRS NOAA-21", "MODIS_NRT": "MODIS",
    }
    successful_sources = [source for source in FIRMS_SOURCES if tile_results[source]]
    status_parts = []
    for source in successful_sources:
        label = source_labels.get(source, source)
        count = len(tile_results[source])
        status_parts.append(label if count == len(tiles) else f"{label} ({count}/{len(tiles)} alan)")
    status = "NASA FIRMS Türkiye • " + ", ".join(status_parts)
    if failed:
        status += f" • {len(failed)} alt alan alınamadı; kapsam kısmi"
    return result, status


def fetch_nasa_firms_extent_hotspots(
    bbox: tuple[float, float, float, float],
    map_key: str | None = None,
    days: int = 1,
) -> tuple[pd.DataFrame, str]:
    """Return cached FIRMS records for the Turkey extent; never synthesize data."""
    try:
        return _fetch_firms_extent_success(bbox, map_key or "", days)
    except FirmsUnavailable as exc:
        return pd.DataFrame(columns=[*FIRMS_COLUMNS, "is_demo"]), str(exc)


def get_firms_hotspots(
    bbox: tuple[float, float, float, float], map_key: str, allow_demo: bool = False
) -> tuple[pd.DataFrame, str]:
    """Return cached successful FIRMS data; convert uncached errors to UI state."""
    try:
        return _fetch_firms_success(bbox, map_key)
    except FirmsUnavailable as exc:
        return _firms_error(str(exc), bbox, allow_demo)


def clear_firms_hotspots_cache() -> None:
    """Clear only successful FIRMS responses, never Esri/vision/geocode caches."""
    _fetch_firms_success.clear()


def fetch_nasa_firms_hotspots(
    bbox: tuple[float, float, float, float],
    map_key: str | None = None,
    allow_demo: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Named entry point for the NASA FIRMS data source."""
    return get_firms_hotspots(bbox, map_key or "", allow_demo)


def anomaly_priority(row: pd.Series | Any) -> tuple[int, str, str]:
    """Transparent 0–100 triage score, not a probability of fire."""
    get = row.get if hasattr(row, "get") else lambda key, default=None: getattr(row, key, default)
    parts: list[tuple[float, float]] = []
    raw_confidence = get("confidence", "")
    confidence = str(raw_confidence).strip().lower()
    confidence_value = {"h": 100.0, "high": 100.0, "n": 65.0, "nominal": 65.0, "l": 35.0, "low": 35.0}.get(confidence)
    # MODIS FIRMS expresses confidence as a percentage, while VIIRS commonly
    # uses h/n/l categories. NASA MODIS classes: <30 low, 30–<80 nominal,
    # and >=80 high. This keeps MODIS/Terra/Aqua in the same scoring path.
    if confidence_value is None:
        numeric_confidence = pd.to_numeric(raw_confidence, errors="coerce")
        if pd.notna(numeric_confidence) and 0 <= float(numeric_confidence) <= 100:
            confidence_value = 100.0 if numeric_confidence >= 80 else 65.0 if numeric_confidence >= 30 else 35.0
    if confidence_value is not None:
        parts.append((0.45, confidence_value))
    frp = pd.to_numeric(get("frp"), errors="coerce")
    if pd.notna(frp) and float(frp) >= 0:
        parts.append((0.35, 100.0 * float(frp) / (float(frp) + 10.0)))
    ti4 = pd.to_numeric(get("bright_ti4"), errors="coerce")
    if pd.notna(ti4):
        parts.append((0.20, max(0.0, min(100.0, (float(ti4) - 300.0) / 0.6))))
    if not parts:
        return 0, "Puanlanamadı", "#68746d"
    score = int(round(sum(weight * value for weight, value in parts) / sum(w for w, _ in parts)))
    if score >= 75:
        return score, "Yüksek inceleme önceliği", "#bd342b"
    if score >= 50:
        return score, "Orta inceleme önceliği", "#d77b20"
    return score, "Düşük inceleme önceliği", "#b79a35"


def secret_or_environment(name: str) -> str:
    try:
        value: Any = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or os.environ.get(name, "")).strip()
