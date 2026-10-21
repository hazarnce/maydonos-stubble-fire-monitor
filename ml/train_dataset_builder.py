"""Build an auditable 2020–2025 Turkey FIRMS/control dataset.

FIRMS is an observed thermal-hotspot proxy, not a label for illegal stubble
burning. Negative controls mean no reported FIRMS detection or field-checked
clear, depending on their `review_status` value. They are never called proven
fire-free unless independently field verified.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests

from ml.feature_engineering import (
    MODEL_FEATURES,
    PERSISTENCE_GRID_SIZE_M,
    PERSISTENCE_SENSOR_SET,
    add_hotspot_persistence_features,
    calculate_ffmc_features,
    normalize_firms_sensor,
    normalize_firms_sensors,
    prepare_persistence_events,
)
from ml.harvest_calendar import (
    HARVEST_MODEL_FEATURES,
    HARVEST_SOURCES,
    REGIONAL_HARVEST_WINDOWS,
    REGIONS,
    annotate_harvest_context,
    harvest_context,
    harvest_model_features,
    is_in_harvest_window,
    province_at,
)

TURKEY_BBOX = (25.6, 35.8, 44.8, 42.2)  # west, south, east, north
FIRMS_DIR = Path("data/raw/firms")
CONTROLS_PATH = Path("data/raw/control_points.csv")
OUTPUT_PATH = Path("data/processed/turkey_stubble_fire_dataset.csv")
DEFAULT_CORINE_RASTER = Path("data/raw/clc2018_raster100m.tif")
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
CORINE_QUERY_URL = (
    "https://copernicus.discomap.eea.europa.eu/arcgis/rest/services/"
    "Corine/CLC2018_LAEA/MapServer/0/query"
)
WEATHER_VARS = (
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation",
    "vapour_pressure_deficit", "soil_moisture_0_to_7cm",
)
# Copernicus CLC categorical rasters may store 1-based class indices rather
# than three-digit codes. Keep this ordered mapping aligned with the official
# 44-class CLC legend.
CLC_INDEX_TO_CODE = dict(enumerate([
    111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142,
    211, 212, 213, 221, 222, 223, 231, 241, 242, 243, 244,
    311, 312, 313, 321, 322, 323, 324, 331, 332, 333, 334, 335,
    411, 412, 421, 422, 423, 511, 512, 521, 522, 523,
], start=1))
VALID_CLC_CODES = {str(code) for code in CLC_INDEX_TO_CODE.values()}
CORINE_PROFILES = {
    # Main target: annual/seasonal arable fields and complex cultivation.
    "grain": {"211", "212", "241", "242", "243"},
    # Sensitivity only: all CLC agricultural classes; not the primary stubble target.
    "all_agriculture": {
    "211", "212", "213", "221", "222", "223", "231",
    "241", "242", "243", "244",
    },
}
MATCHED_CONTROL_CORINE_CODES = CORINE_PROFILES["grain"]
DIYARBAKIR_QA_BBOX = (39.5, 37.5, 41.0, 38.5)  # west, south, east, north
CORINE_NO_FEATURE = "__NO_FEATURE__"
LANDUSE_NAMES = {0: "other", 1: "farmland", 2: "forest", 3: "industrial"}
LANDUSE_TEXT = {value: key for key, value in LANDUSE_NAMES.items()}


def _inside_turkey_bbox(frame: pd.DataFrame, lat_col: str, lon_col: str) -> pd.Series:
    west, south, east, north = TURKEY_BBOX
    return frame[lon_col].between(west, east) & frame[lat_col].between(south, north)


def _read_firms_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        # FIRMS archive ZIPs include a Readme.txt beside the CSV, so pandas'
        # compression="zip" shortcut rejects them as ambiguous. Read only CSV
        # members and combine them if the archive contains multiple partitions.
        with ZipFile(path) as archive:
            csv_members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if not csv_members:
                raise ValueError(f"{path.name}: ZIP içinde CSV bulunamadı.")
            frames = [pd.read_csv(archive.open(name), low_memory=False) for name in csv_members]
        return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return pd.read_csv(path, low_memory=False)


def load_firms_archive(firms_dir: Path) -> pd.DataFrame:
    files = sorted([*firms_dir.glob("*.csv"), *firms_dir.glob("*.zip")])
    if not files:
        raise FileNotFoundError(
            f"NASA FIRMS arşivi bulunamadı: {firms_dir}. 2020–2025 VIIRS arşiv CSV/ZIP dosyalarını bu klasöre koyun."
        )
    frames: list[pd.DataFrame] = []
    for path in files:
        print(f"FIRMS arşivi okunuyor: {path.name}", flush=True)
        frame = _read_firms_file(path)
        print(f"  CSV yüklendi: {len(frame):,} satır", flush=True)
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        if "bright_ti4" not in frame and "brightness" in frame:
            frame["bright_ti4"] = frame["brightness"]
        required = {"latitude", "longitude", "acq_date", "acq_time"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path.name}: NASA FIRMS için gerekli alanlar yok: {sorted(required - set(frame.columns))}")
        for col in ("latitude", "longitude", "frp", "bright_ti4"):
            if col in frame:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if "frp" not in frame:
            frame["frp"] = np.nan
        if "bright_ti4" not in frame:
            frame["bright_ti4"] = np.nan
        frame["acq_date"] = pd.to_datetime(frame["acq_date"], errors="coerce").dt.date
        hhmm = pd.to_numeric(frame["acq_time"], errors="coerce")
        frame["acq_hour"] = (hhmm // 100).astype("Int64")
        frame["acq_time"] = hhmm
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result = result.dropna(subset=["latitude", "longitude", "acq_date", "acq_hour"])
    result = result.loc[_inside_turkey_bbox(result, "latitude", "longitude")].copy()
    dates = pd.to_datetime(result["acq_date"])
    result = result.loc[dates.dt.year.between(2020, 2025)].copy()
    result["_model_sensor"] = normalize_firms_sensors(result)
    result = result.loc[result["_model_sensor"].isin(PERSISTENCE_SENSOR_SET)].drop(
        columns=["_model_sensor"]
    ).copy()
    if result.empty:
        raise ValueError("Eğitim arşivinde S-NPP/NOAA-20 tespiti bulunamadı.")
    before_pixel_dedup = len(result)
    print(f"FIRMS filtreleme tamamlandı: {before_pixel_dedup:,} olay; sensör/piksel tekilleştirme başlıyor", flush=True)
    result = prepare_persistence_events(result)
    print(
        f"Persistence ile aynı sensör ve 375 m × gün tekilleştirmesi: "
        f"{before_pixel_dedup:,} → {len(result):,} arşiv olayı"
    )
    result = annotate_harvest_context(result)
    result["label"] = 1
    result["label_basis"] = "FIRMS_hotspot_proxy"
    result["label_source"] = "NASA_FIRMS_archive"
    result["is_night"] = result.get("daynight", pd.Series("", index=result.index)).astype(str).str.upper().eq("N").astype("Int64")
    result["control_type"] = "hotspot"
    # Collapse exact duplicates across repeated archive files, not nearby pixels.
    keys = ["latitude", "longitude", "acq_date", "acq_time", "satellite"]
    keys = [key for key in keys if key in result.columns]
    return result.drop_duplicates(keys).reset_index(drop=True)


def sample_hotspots_stratified(
    hotspots: pd.DataFrame, max_samples: int, random_seed: int = 42
) -> pd.DataFrame:
    """Bound CORINE work while retaining year, sensor and every 1° CV block."""
    if max_samples < 1:
        raise ValueError("Maksimum örnek sayısı pozitif olmalı.")
    if len(hotspots) <= max_samples:
        return hotspots.copy()
    frame = hotspots.copy().reset_index(drop=True)
    frame["_year"] = pd.to_datetime(frame["acq_date"]).dt.year
    frame["_sensor"] = normalize_firms_sensors(frame)
    # Align sampling strata with the 1° held-out validation groups so rare
    # geographic blocks are not lost before cross-validation.
    frame["_lat_block"] = np.floor(pd.to_numeric(frame["latitude"], errors="coerce"))
    frame["_lon_block"] = np.floor(pd.to_numeric(frame["longitude"], errors="coerce"))
    keys = ["_year", "_sensor", "_lat_block", "_lon_block"]
    grouped = list(frame.groupby(keys, sort=True, dropna=False).groups.items())
    if max_samples < len(grouped):
        raise ValueError(
            f"Örnek üst sınırı {max_samples}, fakat yılı/sensörü/1° bölgeyi korumak için "
            f"en az {len(grouped)} gerekir. --max-event-samples değerini artırın."
        )
    sizes = np.asarray([len(indices) for _, indices in grouped], dtype=int)
    ideal = sizes * (max_samples / sizes.sum())
    quota = np.maximum(1, np.floor(ideal).astype(int))
    while quota.sum() > max_samples:
        candidates = np.flatnonzero(quota > 1)
        index = candidates[np.argmax(quota[candidates] - ideal[candidates])]
        quota[index] -= 1
    remainder_order = np.argsort(-(ideal - np.floor(ideal)))
    cursor = 0
    while quota.sum() < max_samples:
        index = remainder_order[cursor % len(remainder_order)]
        if quota[index] < sizes[index]:
            quota[index] += 1
        cursor += 1
    rng = np.random.default_rng(random_seed)
    selected: list[int] = []
    for (_, indices), count in zip(grouped, quota):
        values = np.asarray(list(indices), dtype=int)
        selected.extend(rng.choice(values, size=int(count), replace=False).tolist())
    return frame.loc[sorted(selected)].drop(columns=keys).reset_index(drop=True)


def generate_no_detection_controls(
    hotspots: pd.DataFrame,
    *,
    occupancy_events: pd.DataFrame | None = None,
    cell_size: float = 0.1,
    random_seed: int = 42,
    matched_corine_codes: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample no-detection days at random, CORINE-matched positive locations.

    Generated controls are restricted to CORINE agricultural classes. A
    control's coordinate is randomly sampled from a positive hotspot location
    in the same grid cell and exact CORINE class; dates are selected only from
    days with no archived detection in that cell. These are proxy controls, not
    field-verified fire-free observations.
    """
    if hotspots.empty:
        raise ValueError("Kontrol üretmek için FIRMS sıcak nokta kaydı yok.")
    if "corine_code" not in hotspots:
        raise ValueError("Eşleştirilmiş kontrol için pozitif FIRMS satırlarına CORINE kodu eklenmeli.")
    if "harvest_region_code" not in hotspots:
        raise ValueError("Kontrol eşlemesi için pozitiflere bölgesel hasat takvimi eklenmeli.")
    if not math.isfinite(cell_size) or not 0.02 <= cell_size <= 0.25:
        raise ValueError("Kontrol hücre boyutu 0.02–0.25 derece arasında olmalı.")

    west, south, east, north = TURKEY_BBOX
    rows = hotspots.copy()
    matched_corine_codes = matched_corine_codes or MATCHED_CONTROL_CORINE_CODES
    rows["corine_code"] = rows["corine_code"].astype(str).str.replace(r"\.0$", "", regex=True)
    rows = rows.loc[rows["corine_code"].isin(matched_corine_codes)].copy()
    if rows.empty:
        raise ValueError(
            "Eşleştirilmiş kontrol üretilemedi: pozitif FIRMS kayıtlarında CORINE tarım sınıfı (2xx) yok."
        )
    rows["_cell_x"] = np.floor((rows["longitude"] - west) / cell_size).astype(int)
    rows["_cell_y"] = np.floor((rows["latitude"] - south) / cell_size).astype(int)
    positives = rows.groupby(
        ["_cell_y", "_cell_x", "acq_date", "corine_code", "harvest_region_code"], as_index=False
    ).agg(
        latitude=("latitude", "first"), longitude=("longitude", "first"),
        acq_time=("acq_time", "min"), frp=("frp", "max"),
        bright_ti4=("bright_ti4", "max"), is_night=("is_night", "max"),
        source_detection_count=("acq_date", "size"),
        landuse_code=("landuse_code", "first"),
        province_name=("province_name", "first"),
        harvest_region=("harvest_region", "first"),
        harvest_start=("harvest_start", "first"),
        harvest_end=("harvest_end", "first"),
        harvest_progress=("harvest_progress", "mean"),
        **{feature: (feature, "first") for feature in HARVEST_MODEL_FEATURES if feature != "harvest_progress"},
    )
    positives["label"] = 1
    positives["label_basis"] = "FIRMS_hotspot_in_grid_cell_day"
    positives["label_source"] = "NASA_FIRMS_archive"
    positives["control_type"] = "hotspot"

    occupancy = (occupancy_events if occupancy_events is not None else rows).copy()
    occupancy["acq_date"] = pd.to_datetime(occupancy["acq_date"], errors="coerce").dt.date
    occupancy["_cell_x"] = np.floor((occupancy["longitude"] - west) / cell_size).astype(int)
    occupancy["_cell_y"] = np.floor((occupancy["latitude"] - south) / cell_size).astype(int)
    occupied_days = set(
        (int(y), int(x), day)
        for y, x, day in occupancy[["_cell_y", "_cell_x", "acq_date"]].itertuples(index=False, name=None)
    )
    positive_locations = {
        (int(cell_y), int(cell_x), str(code), str(_region)): group[["latitude", "longitude"]].to_numpy(dtype=float)
        for (cell_y, cell_x, code, _region), group in rows.groupby(
            ["_cell_y", "_cell_x", "corine_code", "harvest_region_code"]
        )
    }
    positives["_year"] = pd.to_datetime(positives["acq_date"]).dt.year
    positives["_month"] = pd.to_datetime(positives["acq_date"]).dt.month
    rng = np.random.default_rng(random_seed)
    control_rows: list[dict[str, Any]] = []
    grouped = positives.groupby(
        ["_cell_y", "_cell_x", "_year", "_month", "corine_code", "harvest_region_code"],
        sort=False,
    )
    for (cell_y, cell_x, year, month, corine_code, region_code), positive_group in grouped:
        month_start = pd.Timestamp(year=int(year), month=int(month), day=1)
        month_days = pd.date_range(
            month_start, periods=calendar.monthrange(int(year), int(month))[1], freq="D"
        ).date
        occupied = occupied_days.intersection(
            (int(cell_y), int(cell_x), day) for day in month_days
        )
        available = [
            day for day in month_days
            if (int(cell_y), int(cell_x), day) not in occupied
            and is_in_harvest_window(day, str(region_code))
        ]
        count = min(len(positive_group), len(available))
        if count <= 0:
            continue
        chosen_days = rng.choice(np.asarray(available, dtype=object), size=count, replace=False)
        candidate_locations = positive_locations[
            (int(cell_y), int(cell_x), str(corine_code), str(region_code))
        ]
        for day in chosen_days:
            sampled_location = candidate_locations[int(rng.integers(0, len(candidate_locations)))]
            province = province_at(float(sampled_location[0]), float(sampled_location[1]))
            context = harvest_context(day, province)
            harvest_features = harvest_model_features(day, str(region_code))
            row = {
                "latitude": float(sampled_location[0]),
                "longitude": float(sampled_location[1]),
                "acq_date": day, "acq_time": np.nan, "frp": np.nan,
                "bright_ti4": np.nan, "is_night": pd.NA,
                "source_detection_count": 0, "label": 0,
                "label_basis": "no_FIRMS_detection_in_matched_cell_class_day",
                "label_source": "NASA_FIRMS_archive_sampled_absence",
                "control_type": "no_firms_detection_control",
                "corine_code": str(corine_code),
                "landuse_code": 1,
                "province_name": province,
                "harvest_region": REGIONS[str(region_code)],
                "harvest_region_code": str(region_code),
                "harvest_start": context.get("window_start"),
                "harvest_end": context.get("window_end"),
                "_cell_y": int(cell_y), "_cell_x": int(cell_x),
            }
            row.update(harvest_features)
            control_rows.append(row)
    if not control_rows:
        raise ValueError("CORINE eşleşmeli, arşivde tespitsiz kontrol günü bulunamadı.")
    controls = pd.DataFrame(control_rows)
    positives = positives.drop(columns=["_cell_x", "_cell_y", "_year", "_month"])
    controls = controls.drop(columns=["_cell_x", "_cell_y"])
    return positives.reset_index(drop=True), controls.reset_index(drop=True)

def load_control_points(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Kontrol noktaları bulunamadı: {path}. Şablon için `data/raw/control_points_template.csv` dosyasına bakın."
        )
    controls = pd.read_csv(path, low_memory=False)
    controls.columns = [str(c).strip().lower() for c in controls.columns]
    required = {"latitude", "longitude", "date", "landuse_type", "review_status"}
    missing = required.difference(controls.columns)
    if missing:
        raise ValueError(f"Kontrol CSV'sinde gerekli sütunlar eksik: {', '.join(sorted(missing))}")
    controls["latitude"] = pd.to_numeric(controls["latitude"], errors="coerce")
    controls["longitude"] = pd.to_numeric(controls["longitude"], errors="coerce")
    controls["acq_date"] = pd.to_datetime(controls["date"], errors="coerce").dt.date
    if "acq_time" in controls:
        controls["acq_time"] = pd.to_numeric(controls["acq_time"], errors="coerce")
    else:
        controls["acq_time"] = np.nan
    controls = controls.dropna(subset=["latitude", "longitude", "acq_date"]).copy()
    controls = controls.loc[_inside_turkey_bbox(controls, "latitude", "longitude")].copy()
    dates = pd.to_datetime(controls["acq_date"])
    controls = controls.loc[dates.dt.year.between(2020, 2025)].copy()
    controls = annotate_harvest_context(controls)
    seasonal_controls = controls.loc[controls["in_harvest_window"]].copy()
    controls.attrs["harvest_controls_excluded_outside_window"] = int(len(controls) - len(seasonal_controls))
    controls = seasonal_controls
    if controls.empty:
        raise ValueError("Kullanıcı kontrol kayıtlarının hiçbiri bölgesel hububat hasat penceresine girmiyor.")
    controls["landuse_type"] = controls["landuse_type"].astype(str).str.strip().str.lower()
    if not controls["landuse_type"].isin(LANDUSE_TEXT).all():
        invalid = sorted(controls.loc[~controls["landuse_type"].isin(LANDUSE_TEXT), "landuse_type"].unique())
        raise ValueError(f"landuse_type değerleri farmland/forest/industrial/other olmalı; hatalı: {invalid}")
    allowed_status = {"field_verified_clear", "no_firms_detection_control"}
    controls["review_status"] = controls["review_status"].astype(str).str.strip().str.lower()
    if not controls["review_status"].isin(allowed_status).all():
        raise ValueError("review_status yalnızca field_verified_clear veya no_firms_detection_control olabilir.")
    controls["label"] = 0
    controls["label_basis"] = controls["review_status"]
    controls["label_source"] = "user_control_points"
    controls["is_night"] = pd.NA  # kept missing; not a classifier feature
    controls["frp"] = np.nan
    controls["bright_ti4"] = np.nan
    controls["control_type"] = controls["landuse_type"]
    controls["landuse_code"] = controls["landuse_type"].map(LANDUSE_TEXT)
    return controls.reset_index(drop=True)


def remove_controls_near_hotspots(
    controls: pd.DataFrame, hotspots: pd.DataFrame, radius_m: float = 1000.0
) -> tuple[pd.DataFrame, int]:
    """Discard controls close to a same-day FIRMS detection (not proof of no fire)."""
    if controls.empty or hotspots.empty:
        return controls.copy(), 0
    from sklearn.neighbors import BallTree

    fire_by_day = {
        day: group[["latitude", "longitude"]].to_numpy(dtype=float)
        for day, group in hotspots.groupby("acq_date", sort=False)
    }
    kept: list[int] = []
    excluded = 0
    earth_radius = 6_371_000.0
    for day, group in controls.groupby("acq_date", sort=False):
        points = fire_by_day.get(day)
        if points is None or not len(points):
            kept.extend(group.index.tolist())
            continue
        tree = BallTree(np.radians(points), metric="haversine")
        distances, _ = tree.query(
            np.radians(group[["latitude", "longitude"]].to_numpy(dtype=float)), k=1
        )
        keep_mask = distances[:, 0] * earth_radius >= radius_m
        kept.extend(group.index[keep_mask].tolist())
        excluded += int((~keep_mask).sum())
    return controls.loc[kept].reset_index(drop=True), excluded


def _classify_corine(code: str | None) -> int | None:
    if not code:
        return None
    try:
        code_i = int(str(code).split(".")[0])
    except ValueError:
        return None
    if 200 <= code_i < 300:
        return 1
    if 300 <= code_i < 400:
        return 2
    if code_i in {121, 131, 132, 133}:
        return 3
    return 0


def fetch_corine_codes(
    samples: pd.DataFrame,
    cache_path: Path,
    *,
    local_raster: Path | None = None,
) -> pd.DataFrame:
    """Sample local CLC raster when supplied; otherwise query the official REST layer.

    Point outcomes are counted as feature, successful empty (no polygon), or
    request error. Failed requests are never written to the cache as empty data.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        cache = pd.read_csv(cache_path)
    else:
        cache = pd.DataFrame(columns=["lat_key", "lon_key", "corine_code", "corine_name"])
    has_status = "lookup_status" in cache.columns
    cache_map: dict[tuple[float, float], tuple[str, str, str]] = {}
    for row in cache.itertuples():
        if pd.isna(row.corine_code) or not str(row.corine_code).strip():
            continue
        code = str(row.corine_code)
        status = str(getattr(row, "lookup_status", "")) if has_status else "legacy"
        # Old sentinel values conflated a successful empty query with a failed
        # request. Re-query only those ambiguous rows; actual CLC codes are safe.
        if code == CORINE_NO_FEATURE and status not in {"empty", "feature"}:
            continue
        cache_map[(round(float(row.lat_key), 5), round(float(row.lon_key), 5))] = (
            code, str(row.corine_name), status if status != "legacy" else "feature"
        )
    locations = sorted({(round(float(r.latitude), 5), round(float(r.longitude), 5)) for r in samples.itertuples()})
    missing = [point for point in locations if point not in cache_map]

    lookup_stats = {
        "unique_locations": int(len(locations)),
        "cache_hits": int(len(locations) - len(missing)),
        "successful_query_with_polygon": 0,
        "successful_empty_no_polygon": 0,
        "exception_or_http_error": 0,
        "request_attempts_succeeded": 0,
        "request_attempts_failed": 0,
        "retry_batches": 0,
        "source": "local_raster" if local_raster else "Copernicus ArcGIS REST",
    }

    if local_raster:
        if not local_raster.exists():
            raise FileNotFoundError(f"CLC 2018 raster bulunamadı: {local_raster}")
        try:
            import rasterio
            from rasterio.warp import transform as raster_transform
        except ImportError as exc:
            raise RuntimeError("Yerel CLC raster örneklemesi için rasterio kurun: pip install rasterio") from exc
        raster_value_counts: dict[str, int] = {}
        with rasterio.open(local_raster) as src:
            if src.crs is None:
                raise ValueError("CLC raster CRS bilgisi taşımıyor; koordinatlar güvenle dönüştürülemez.")
            valid_samples = 0
            qa_candidate_count = 0
            qa_feature_count = 0
            for start in range(0, len(locations), 20_000):
                batch = locations[start:start + 20_000]
                xs, ys = raster_transform("EPSG:4326", src.crs, [p[1] for p in batch], [p[0] for p in batch])
                for point, sample in zip(batch, src.sample(zip(xs, ys), indexes=1, masked=True)):
                    west, south, east, north = DIYARBAKIR_QA_BBOX
                    in_qa_area = west <= point[1] <= east and south <= point[0] <= north
                    if in_qa_area:
                        qa_candidate_count += 1
                    value = sample[0]
                    if np.ma.is_masked(value) or not np.isfinite(float(value)):
                        cache_map[point] = (CORINE_NO_FEATURE, CORINE_NO_FEATURE, "empty")
                        continue
                    iv = int(value)
                    raster_value_counts[str(iv)] = raster_value_counts.get(str(iv), 0) + 1
                    if iv in CLC_INDEX_TO_CODE:
                        code = str(CLC_INDEX_TO_CODE[iv])
                    elif str(iv) in VALID_CLC_CODES:
                        # Support raster products that store the official code directly.
                        code = str(iv)
                    else:
                        # Includes 0, 255/128 NODATA sentinels and unknown classes.
                        cache_map[point] = (CORINE_NO_FEATURE, CORINE_NO_FEATURE, "empty")
                        continue
                    valid_samples += 1
                    if in_qa_area:
                        qa_feature_count += 1
                    cache_map[point] = (code, code, "feature")
            qa_feature_ratio = qa_feature_count / qa_candidate_count if qa_candidate_count else None
            if qa_candidate_count >= 100 and qa_feature_ratio is not None and qa_feature_ratio < 0.90:
                raise RuntimeError(
                    "CLC raster sağlık kontrolü başarısız: Diyarbakır örneklerinde geçerli sınıf oranı "
                    f"%{qa_feature_ratio * 100:.1f} (<%90). Raster CRS, nodata ve sınıf legendini doğrulayın."
                )
            qa_points = [
                point for point in locations
                if DIYARBAKIR_QA_BBOX[0] <= point[1] <= DIYARBAKIR_QA_BBOX[2]
                and DIYARBAKIR_QA_BBOX[1] <= point[0] <= DIYARBAKIR_QA_BBOX[3]
            ]
            qa_agriculture_count = sum(
                1 for point in qa_points
                if cache_map.get(point, (None, None, "error"))[2] == "feature"
                and str(cache_map.get(point, (None, None, "error"))[0]).startswith("2")
            )
            qa_agriculture_ratio = (
                qa_agriculture_count / qa_feature_count if qa_feature_count else None
            )
            if (
                qa_candidate_count >= 100
                and qa_feature_count
                and qa_agriculture_ratio is not None
                and qa_agriculture_ratio < 0.50
            ):
                raise RuntimeError(
                    "CLC eşleme şüpheli: Diyarbakır örneklerinde 2xx tarım/mera sınıf oranı "
                    f"%{qa_agriculture_ratio * 100:.1f} (<%50). Eşleme tablosunu ve raster legendini kontrol edin."
                )
            if qa_candidate_count < 100:
                print(f"CLC sağlık kontrolü: Diyarbakır BBOX içinde {qa_candidate_count} örnek var; 100 örnek altı, oran testi atlandı.")
            print(
                "CLC raster QA: "
                f"örnek={len(locations):,}, geçerli-sınıf={valid_samples:,}, "
                f"Diyarbakır-örnek={qa_candidate_count:,}, 2xx-oranı="
                f"{qa_agriculture_ratio if qa_agriculture_ratio is not None else 'n/a'}, "
                f"sınıf-değerleri={raster_value_counts}"
            )
        lookup_stats["raster_value_counts"] = raster_value_counts
        lookup_stats["diyarbakir_qa_samples"] = qa_candidate_count
        lookup_stats["diyarbakir_qa_valid_feature_ratio"] = qa_feature_ratio
        lookup_stats["diyarbakir_qa_agriculture_class_count"] = qa_agriculture_count
        lookup_stats["diyarbakir_qa_agriculture_class_ratio"] = qa_agriculture_ratio
        lookup_stats["diyarbakir_qa_agriculture_ratio_threshold"] = 0.50
        lookup_stats["successful_query_with_polygon"] = sum(cache_map.get(p, (None, None, "error"))[2] == "feature" for p in locations)
        lookup_stats["successful_empty_no_polygon"] = sum(cache_map.get(p, (None, None, "error"))[2] == "empty" for p in locations)
        lookup_stats["exception_or_http_error"] = 0
        cache = pd.DataFrame([
            {"lat_key": p[0], "lon_key": p[1], "corine_code": v[0], "corine_name": v[1], "lookup_status": v[2]}
            for p, v in cache_map.items()
        ])
        cache.to_csv(cache_path, index=False)

    def lookup_batch(points: list[tuple[float, float]]) -> list[tuple[tuple[float, float], str | None, str | None, str]]:
        """Look up many points in one ArcGIS request, then spatially join polygons."""
        from shapely.geometry import Point, Polygon

        geometry = {
            "points": [[lon, lat] for lat, lon in points],
            "spatialReference": {"wkid": 4326},
        }
        params = {
            "geometry": json.dumps(geometry, separators=(",", ":")),
            "geometryType": "esriGeometryMultipoint", "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects", "outFields": "Code_18",
            "returnGeometry": "true", "outSR": "4326", "resultRecordCount": 2000, "f": "json",
        }
        response = requests.get(CORINE_QUERY_URL, params=params, timeout=(8, 45))
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise ValueError(str(payload["error"].get("message", "CORINE hata")))
        if payload.get("exceededTransferLimit"):
            raise ValueError("CORINE toplu sorgusu aktarım sınırını aştı; küçük paketle yeniden deneyin.")

        polygons: list[tuple[Any, str]] = []

        def signed_area(ring: list[list[float]]) -> float:
            return sum(
                ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
                for i in range(len(ring) - 1)
            ) / 2.0

        for feature in payload.get("features", []):
            code = str(feature.get("attributes", {}).get("Code_18", "")).strip()
            rings = (feature.get("geometry") or {}).get("rings") or []
            shells = [ring for ring in rings if signed_area(ring) < 0]
            holes = [ring for ring in rings if signed_area(ring) >= 0]
            shell_polygons = [(ring, Polygon(ring)) for ring in shells]
            for shell, shell_polygon in shell_polygons:
                assigned_holes = []
                for ring in holes:
                    hole = Polygon(ring)
                    if shell_polygon.covers(hole.representative_point()):
                        assigned_holes.append(ring)
                polygon = Polygon(shell, assigned_holes)
                if not polygon.is_empty and polygon.is_valid:
                    polygons.append((polygon, code))

        answers: list[tuple[tuple[float, float], str | None, str | None, str]] = []
        for point in points:
            lat, lon = point
            candidate_codes = [
                code for polygon, code in polygons
                if polygon.bounds[0] <= lon <= polygon.bounds[2]
                and polygon.bounds[1] <= lat <= polygon.bounds[3]
                and polygon.covers(Point(lon, lat))
            ]
            code = candidate_codes[0] if candidate_codes else None
            answers.append((point, code, code, "feature" if code else "empty"))
        return answers

    if missing and not local_raster:
        batch_size = 16
        batches = [missing[index:index + batch_size] for index in range(0, len(missing), batch_size)]
        print(f"CORINE: {len(missing)} koordinat {len(batches)} toplu sorguda işleniyor; önbellekte {len(cache_map)} kayıt var.")
        additions: list[dict[str, Any]] = []

        def lookup_with_fallback(
            points: list[tuple[float, float]],
        ) -> tuple[list[tuple[tuple[float, float], str | None, str | None, str]], dict[str, int]]:
            """Retry once, then split failed batches; terminal failures remain uncached."""
            stats = {"request_attempts_succeeded": 0, "request_attempts_failed": 0, "retry_batches": 0}
            try:
                answers = lookup_batch(points)
                stats["request_attempts_succeeded"] += 1
                return answers, stats
            except (requests.RequestException, ValueError, TypeError, KeyError):
                stats["request_attempts_failed"] += 1
                for retry_index in range(2):
                    stats["retry_batches"] += 1
                    time.sleep(0.5 * (2 ** retry_index))
                    try:
                        answers = lookup_batch(points)
                        stats["request_attempts_succeeded"] += 1
                        return answers, stats
                    except (requests.RequestException, ValueError, TypeError, KeyError):
                        stats["request_attempts_failed"] += 1
                if len(points) == 1:
                    return [(points[0], None, None, "error")], stats
                midpoint = len(points) // 2
                left, left_stats = lookup_with_fallback(points[:midpoint])
                right, right_stats = lookup_with_fallback(points[midpoint:])
                for key in stats:
                    stats[key] += left_stats[key] + right_stats[key]
                return left + right, stats

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(lookup_with_fallback, batch): batch for batch in batches}
            for number, future in enumerate(as_completed(futures), start=1):
                batch_results, request_stats = future.result()
                for key, value in request_stats.items():
                    lookup_stats[key] += value
                for point, code, name, status in batch_results:
                    if status == "error":
                        lookup_stats["exception_or_http_error"] += 1
                        continue  # Never turn an HTTP failure into a cached empty result.
                    if status == "feature":
                        lookup_stats["successful_query_with_polygon"] += 1
                    else:
                        lookup_stats["successful_empty_no_polygon"] += 1
                    stored_code = code if code else CORINE_NO_FEATURE
                    stored_name = (name or code) if code else CORINE_NO_FEATURE
                    cache_map[point] = (stored_code, stored_name, status)
                    additions.append({"lat_key": point[0], "lon_key": point[1], "corine_code": stored_code, "corine_name": stored_name, "lookup_status": status})
                if number % 5 == 0 or number == len(batches):
                    print(f"CORINE toplu sorgusu: {number}/{len(batches)}")
                # Keep completed API work if a later request is interrupted.
                if additions and (number % 10 == 0 or number == len(batches)):
                    cache = pd.concat([cache, pd.DataFrame(additions)], ignore_index=True)
                    cache = cache.drop_duplicates(["lat_key", "lon_key"], keep="last")
                    cache.to_csv(cache_path, index=False)
                    additions.clear()
        if additions:
            cache = pd.concat([cache, pd.DataFrame(additions)], ignore_index=True).drop_duplicates(["lat_key", "lon_key"], keep="last")
            cache.to_csv(cache_path, index=False)
    result = samples.copy()
    keys = [(round(float(r.latitude), 5), round(float(r.longitude), 5)) for r in result.itertuples()]
    raw_codes = [cache_map.get(key, (None, None, "error"))[0] for key in keys]
    result["corine_lookup_status"] = [cache_map.get(key, (None, None, "error"))[2] for key in keys]
    result["corine_code"] = [None if code == CORINE_NO_FEATURE else code for code in raw_codes]
    result["landuse_code"] = [
        _classify_corine(cache_map.get(key, (None, None, "error"))[0]) for key in keys
    ]
    matched = result.get("corine_code", pd.Series(index=result.index, dtype=object)).notna().sum()
    no_feature = int(result["corine_code"].isna().sum())
    status_by_location = {point: cache_map.get(point, (None, None, "error"))[2] for point in locations}
    lookup_stats["successful_query_with_polygon"] = sum(status == "feature" for status in status_by_location.values())
    lookup_stats["successful_empty_no_polygon"] = sum(status == "empty" for status in status_by_location.values())
    lookup_stats["exception_or_http_error"] = sum(status == "error" for status in status_by_location.values())
    print(
        f"CORINE sonuç: poligonlu={lookup_stats['successful_query_with_polygon']:,}, "
        f"boş yanıt={lookup_stats['successful_empty_no_polygon']:,}, "
        f"HTTP/istisna={lookup_stats['exception_or_http_error']:,}, "
        f"başarılı istek={lookup_stats['request_attempts_succeeded']:,}, "
        f"başarısız istek={lookup_stats['request_attempts_failed']:,}, "
        f"yeniden deneme={lookup_stats['retry_batches']:,}"
    )
    result.attrs["corine_lookup_stats"] = lookup_stats
    return result


def fetch_weather_features(samples: pd.DataFrame, cache_dir: Path, chunk_size: int = 20) -> pd.DataFrame:
    """Fetch hourly ERA5 weather for pre-event day at 12 UTC; sum its prior 24h rain.

    ERA5 (~0.25°) is selected for consistent 2020–2025 coverage of all requested
    variables, including VPD and topsoil moisture. Reanalysis is not a station observation. Hourly coordinates are
    snapped to a 0.25° sampling grid to reduce redundant requests.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = samples.copy()
    result["weather_lat"] = (result["latitude"].astype(float) * 4).round() / 4
    result["weather_lon"] = (result["longitude"].astype(float) * 4).round() / 4
    result["target_date"] = pd.to_datetime(result["acq_date"], errors="coerce")
    result["weather_date"] = result["target_date"] - pd.Timedelta(days=1)
    result["weather_year"] = result["weather_date"].dt.year
    result["weather_month"] = result["weather_date"].dt.month
    if result[["weather_lat", "weather_lon", "weather_date"]].isna().any().any():
        raise ValueError("Hava servisine gönderilecek konum/tarih eksik.")

    weather_parts: list[pd.DataFrame] = []

    def fetch_batch(
        batch: list[tuple[float, float]], cache_file: Path,
        start_text: str, end_text: str, year: int, month: int,
    ) -> pd.DataFrame:
        params = {
            "latitude": ",".join(f"{lat:.2f}" for lat, _ in batch),
            "longitude": ",".join(f"{lon:.2f}" for _, lon in batch),
            "start_date": start_text, "end_date": end_text,
            "hourly": ",".join(WEATHER_VARS), "timezone": "GMT",
            "wind_speed_unit": "kmh", "precipitation_unit": "mm", "models": "era5",
        }
        response = None
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                response = requests.get(WEATHER_URL, params=params, timeout=(10, 120))
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                last_error = exc
                if attempt < 4:
                    time.sleep(min(2 ** attempt, 12))
        if response is None:
            raise RuntimeError(f"Open-Meteo arşiv verisi alınamadı ({year}-{month:02d}): {last_error}")
        payload = response.json()
        locations = payload if isinstance(payload, list) else [payload]
        if len(locations) != len(batch):
            raise ValueError("Open-Meteo çoklu-konum yanıtı beklenen konum sayısıyla eşleşmiyor.")
        records: list[dict[str, Any]] = []
        for (lat, lon), location in zip(batch, locations):
            hourly = location.get("hourly") or {}
            times = hourly.get("time") or []
            arrays = {name: hourly.get(name) or [] for name in WEATHER_VARS}
            if any(len(values) != len(times) for values in arrays.values()):
                continue
            for i, timestamp in enumerate(times):
                records.append({
                    "weather_lat": lat, "weather_lon": lon,
                    "time": pd.to_datetime(timestamp, utc=True).tz_localize(None),
                    "temp": arrays["temperature_2m"][i],
                    "rh": arrays["relative_humidity_2m"][i],
                    "ws": arrays["wind_speed_10m"][i],
                    "prec_hour": arrays["precipitation"][i],
                    "vapour_pressure_deficit": arrays["vapour_pressure_deficit"][i],
                    "soil_moisture_0_to_7cm": arrays["soil_moisture_0_to_7cm"][i],
                })
        if not records:
            raise ValueError(f"Open-Meteo {year}-{month:02d} yanıtında saatlik değer yok.")
        part = pd.DataFrame(records)
        part.to_csv(cache_file, index=False)
        return part

    for (year, month), month_rows in result.groupby(["weather_year", "weather_month"]):
        coordinates = sorted(set(zip(month_rows["weather_lat"], month_rows["weather_lon"])))
        start = pd.Timestamp(year=int(year), month=int(month), day=1) - pd.Timedelta(days=1)
        end = pd.Timestamp(year=int(year), month=int(month), day=calendar.monthrange(int(year), int(month))[1])
        start_text, end_text = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        pending: dict[Any, Path] = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for offset in range(0, len(coordinates), chunk_size):
                batch = coordinates[offset: offset + chunk_size]
                digest = hashlib.sha256(
                    (f"v2-vpd-soil-{year}-{month}-" + ";".join(f"{lat:.2f},{lon:.2f}" for lat, lon in batch)).encode("utf-8")
                ).hexdigest()[:16]
                cache_file = cache_dir / f"era5_{year}_{month:02d}_{digest}.csv"
                if cache_file.exists():
                    weather_parts.append(pd.read_csv(cache_file, parse_dates=["time"]))
                    continue
                future = pool.submit(fetch_batch, batch, cache_file, start_text, end_text, int(year), int(month))
                pending[future] = cache_file
            for number, future in enumerate(as_completed(pending), start=1):
                weather_parts.append(future.result())
                if number % 5 == 0 or number == len(pending):
                    print(f"Open-Meteo istekleri: {year}-{month:02d} · {number}/{len(pending)}", flush=True)
        print(f"Open-Meteo arşivi işlendi: {year}-{month:02d} · {len(coordinates)} grid noktası")

    weather = pd.concat(weather_parts, ignore_index=True).drop_duplicates(["weather_lat", "weather_lon", "time"])
    weather = weather.sort_values(["weather_lat", "weather_lon", "time"])
    weather["prec"] = weather.groupby(["weather_lat", "weather_lon"], sort=False)["prec_hour"].transform(
        lambda values: values.rolling(24, min_periods=24).sum()
    )
    weather["weather_date"] = weather["time"].dt.normalize()
    weather["hour"] = weather["time"].dt.hour
    noon = weather.loc[
        weather["hour"].eq(12),
        ["weather_lat", "weather_lon", "weather_date", "temp", "rh", "ws", "prec",
         "vapour_pressure_deficit", "soil_moisture_0_to_7cm"],
    ].copy()
    noon = noon.rename(columns={"weather_date": "weather_date_join"}).sort_values(
        ["weather_lat", "weather_lon", "weather_date_join"]
    )
    # Carry the daily FFMC state only across consecutive source dates. When a
    # location has a gap, restart from the documented seed, not a stale state.
    previous_codes: list[float] = []
    for _, group in noon.groupby(["weather_lat", "weather_lon"], sort=False):
        prior_code = 85.0
        prior_date: pd.Timestamp | None = None
        for row in group.itertuples(index=True):
            day = pd.Timestamp(row.weather_date_join)
            if prior_date is None or (day - prior_date).days != 1:
                prior_code = 85.0
            previous_codes.append(prior_code)
            try:
                weather_values = [float(row.temp), float(row.rh), float(row.ws), float(row.prec)]
                if not np.isfinite(weather_values).all() or weather_values[3] < 0:
                    raise ValueError("Eksik/geçersiz günlük hava girdisi")
                prior_code = float(fine_fuel_moisture_code(
                    ffmc_yda=prior_code,
                    temp=weather_values[0],
                    rh=weather_values[1],
                    ws=weather_values[2],
                    prec=weather_values[3],
                ))
                if not math.isfinite(prior_code):
                    raise ValueError("CFFDRS FFMC sonlu bir değer döndürmedi")
                prior_code = float(np.clip(prior_code, 0.0, 99.0))
            except (ValueError, RuntimeError, TypeError, OverflowError):
                prior_code = 85.0
            prior_date = day
    noon["ffmc_yesterday"] = previous_codes
    result["weather_date_join"] = result["weather_date"].dt.normalize()
    result = result.merge(noon, on=["weather_lat", "weather_lon", "weather_date_join"], how="left", validate="many_to_one")
    result["weather_source"] = "Open-Meteo Historical Weather API · ERA5 · önceki gün 12 UTC"
    result["weather_grid_degrees"] = 0.25
    return result


def add_spatial_groups(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    # All years within the same 1° geographic block stay in one held-out fold.
    out["spatial_group"] = (
        np.floor(out["latitude"].astype(float)).astype(int).astype(str)
        + ":" + np.floor(out["longitude"].astype(float)).astype(int).astype(str)
    )
    return out


def build_dataset(
    firms_dir: Path = FIRMS_DIR,
    controls_path: Path = CONTROLS_PATH,
    output_path: Path = OUTPUT_PATH,
    *,
    skip_corine: bool = False,
    skip_weather: bool = False,
    generate_controls: bool = False,
    control_cell_size: float = 0.1,
    max_weather_grid_locations: int | None = 1_000,
    max_event_samples: int | None = 20_000,
    corine_raster: Path | None = None,
    corine_profile: str = "grain",
    random_seed: int = 42,
) -> pd.DataFrame:
    if corine_profile not in CORINE_PROFILES:
        raise ValueError(f"Bilinmeyen CORINE profili: {corine_profile}")
    matched_corine_codes = CORINE_PROFILES[corine_profile]
    archive_events = load_firms_archive(firms_dir)
    raw_archive_rows = len(archive_events)
    regional_events = archive_events.loc[archive_events["in_harvest_window"]].copy().reset_index(drop=True)
    excluded_outside_harvest = int(len(archive_events) - len(regional_events))
    regional_event_counts = {
        code: int(regional_events["harvest_region_code"].eq(code).sum())
        for code in REGIONS
    }
    print(
        f"Bölgesel hasat takvimi: {len(regional_events):,} FIRMS olayı pencere içinde, "
        f"{excluded_outside_harvest:,} olay pencere dışında kaldı."
    )
    funnel: dict[str, int] = {
        "archive_events_after_sensor_pixel_dedup": int(raw_archive_rows),
        "archive_events_inside_regional_harvest_windows": int(len(regional_events)),
        "archive_events_outside_regional_harvest_windows": excluded_outside_harvest,
        "sampled_events_before_corine": 0,
        "corine_any_code": 0,
        "corine_matched_agriculture_classes": 0,
        "corine_successful_query_with_polygon": 0,
        "corine_successful_empty_no_polygon": 0,
        "corine_exception_or_http_error": 0,
        "corine_request_attempts_succeeded": 0,
        "corine_request_attempts_failed": 0,
        "positive_cell_day_rows": 0,
        "controls_generated_before_nearby_filter": 0,
        "controls_removed_near_same_day_hotspot": 0,
        "manual_controls_excluded_outside_regional_harvest_windows": 0,
        "controls_after_nearby_filter": 0,
        "rows_before_weather_grid_cap": 0,
        "rows_removed_by_weather_grid_cap": 0,
        "rows_before_feature_quality_filter": 0,
        "rows_removed_for_missing_required_features": 0,
        "final_positive_rows": 0,
        "final_negative_rows": 0,
    }
    for code, count in regional_event_counts.items():
        funnel[f"harvest_window_events_{code}"] = count
    candidate_hotspots = regional_events
    if max_event_samples is not None and len(candidate_hotspots) > max_event_samples:
        candidate_hotspots = sample_hotspots_stratified(candidate_hotspots, max_event_samples, random_seed)
        print(
            f"CORINE sorgu sınırı için katmanlı FIRMS örneklemesi: "
            f"{len(regional_events):,} -> {len(candidate_hotspots):,} mevsim içi olay "
            f"(yıl × sensör × 1° doğrulama bloğu, seed={random_seed})"
        )
    funnel["sampled_events_before_corine"] = int(len(candidate_hotspots))
    corine_lookup_stats: dict[str, int] = {}
    if generate_controls:
        if skip_corine:
            raise ValueError("--generate-no-detection-controls için CORINE zorunludur; --skip-corine kullanılamaz.")
        raw_hotspots_with_corine = fetch_corine_codes(
            candidate_hotspots, Path("data/cache/corine_training_lookup.csv"),
            local_raster=(corine_raster or DEFAULT_CORINE_RASTER)
            if (corine_raster or DEFAULT_CORINE_RASTER).exists() else None,
        )
        corine_outcomes = raw_hotspots_with_corine.attrs.get("corine_lookup_stats", {})
        funnel["corine_successful_query_with_polygon"] = int(corine_outcomes.get("successful_query_with_polygon", 0))
        funnel["corine_successful_empty_no_polygon"] = int(corine_outcomes.get("successful_empty_no_polygon", 0))
        funnel["corine_exception_or_http_error"] = int(corine_outcomes.get("exception_or_http_error", 0))
        funnel["corine_request_attempts_succeeded"] = int(corine_outcomes.get("request_attempts_succeeded", 0))
        funnel["corine_request_attempts_failed"] = int(corine_outcomes.get("request_attempts_failed", 0))
        lookup_total = max(1, int(corine_outcomes.get("unique_locations", 0)))
        lookup_error_rate = funnel["corine_exception_or_http_error"] / lookup_total
        if lookup_error_rate > 0.10:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            diagnostics = {
                "aborted": True,
                "reason": "CORINE terminal point error rate exceeded 10%; no incomplete training CSV was written.",
                "stage_funnel": funnel,
                "corine_lookup_outcomes": corine_outcomes,
            }
            output_path.with_suffix(".corine-diagnostics.json").write_text(
                json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            raise RuntimeError(
                f"CORINE sorgularının %{lookup_error_rate * 100:.1f}'i başarısız. Eksik arazi bilgisiyle "
                "eğitim verisi yazılmadı; ayrıntı turkey_stubble_fire_dataset.corine-diagnostics.json "
                "dosyasına kaydedildi. Bağlantıyı tekrar deneyin veya CLC 2018 100 m rasterini "
                f"{corine_raster or DEFAULT_CORINE_RASTER} konumuna koyup yeniden çalıştırın."
            )
        matched_training_class = raw_hotspots_with_corine["corine_code"].astype(str).str.replace(
            r"\.0$", "", regex=True
        ).isin(matched_corine_codes)
        corine_lookup_stats = {
            "sampled_positive_points": int(len(raw_hotspots_with_corine)),
            "points_with_any_corine_code": int(raw_hotspots_with_corine["corine_code"].notna().sum()),
            "points_in_matched_control_classes": int(matched_training_class.sum()),
        }
        funnel["corine_any_code"] = corine_lookup_stats["points_with_any_corine_code"]
        funnel["corine_matched_agriculture_classes"] = corine_lookup_stats["points_in_matched_control_classes"]
        print(
            f"Pozitif örnek, CORINE profili '{corine_profile}' "
            f"({', '.join(sorted(matched_corine_codes))}) ile eşleşen: "
            f"{int(matched_training_class.sum()):,}/{len(raw_hotspots_with_corine):,}"
        )
        hotspots, controls = generate_no_detection_controls(
            raw_hotspots_with_corine, occupancy_events=archive_events,
            cell_size=control_cell_size, random_seed=random_seed,
            matched_corine_codes=matched_corine_codes,
        )
        funnel["positive_cell_day_rows"] = int(len(hotspots))
        funnel["controls_generated_before_nearby_filter"] = int(len(controls))
    else:
        hotspots = candidate_hotspots
        controls = load_control_points(controls_path)
        funnel["manual_controls_excluded_outside_regional_harvest_windows"] = int(
            controls.attrs.get("harvest_controls_excluded_outside_window", 0)
        )
    controls, removed = remove_controls_near_hotspots(controls, archive_events)
    funnel["controls_removed_near_same_day_hotspot"] = int(removed)
    funnel["controls_after_nearby_filter"] = int(len(controls))
    print(f"Aynı gün FIRMS sıcak noktasının 1 km yakınındaki kontrol kayıtları çıkarıldı: {removed}")
    samples = pd.concat([hotspots, controls], ignore_index=True, sort=False)
    funnel["rows_before_weather_grid_cap"] = int(len(samples))

    # Keep the archive-derived labels intact, but cap expensive gridded ERA5
    # lookups using a deterministic, spatially spread sample of 0.25° cells.
    # This is an explicit sampling decision, not fabricated weather data.
    selected_weather_cells: list[tuple[float, float]] | None = None
    if max_weather_grid_locations is not None:
        if max_weather_grid_locations < 4:
            raise ValueError("--max-weather-grid-locations en az 4 olmalı; uzamsal test için farklı bölgeler gerekir.")
        samples["_weather_lat"] = (samples["latitude"].astype(float) * 4).round() / 4
        samples["_weather_lon"] = (samples["longitude"].astype(float) * 4).round() / 4
        cells = samples[["_weather_lat", "_weather_lon"]].drop_duplicates().to_numpy()
        available_weather_cells = len(cells)
        if len(cells) > max_weather_grid_locations:
            rng = np.random.default_rng(random_seed)
            chosen = rng.choice(len(cells), size=max_weather_grid_locations, replace=False)
            cells = cells[chosen]
            selected = set((float(lat), float(lon)) for lat, lon in cells)
            keep = [pair in selected for pair in zip(samples["_weather_lat"], samples["_weather_lon"])]
            before_sampling = len(samples)
            samples = samples.loc[keep].copy()
            funnel["rows_removed_by_weather_grid_cap"] = int(before_sampling - len(samples))
            print(
                f"ERA5 örneklemesi: {len(cells)} / {available_weather_cells} "
                f"grid hücresi; seçilmeyen hücre kayıtları çıkarıldı ({before_sampling - len(samples):,})."
            )
        selected_weather_cells = [(float(lat), float(lon)) for lat, lon in cells]
        samples = samples.drop(columns=["_weather_lat", "_weather_lon"])

    # Preserve manually reviewed controls; generated controls are already
    # paired with a randomly sampled positive location in the same CORINE class.
    if "landuse_code" not in samples:
        samples["landuse_code"] = np.nan
    if not skip_corine:
        needs_landcover = samples["landuse_code"].isna()
        if needs_landcover.any():
            enriched = fetch_corine_codes(
                samples.loc[needs_landcover].copy(), Path("data/cache/corine_training_lookup.csv")
            )
            samples.loc[needs_landcover, "landuse_code"] = enriched["landuse_code"].to_numpy()
            if "corine_code" not in samples:
                samples["corine_code"] = pd.Series(pd.NA, index=samples.index, dtype="object")
            samples.loc[needs_landcover, "corine_code"] = enriched["corine_code"].to_numpy()

    if not skip_weather:
        samples = fetch_weather_features(samples, Path("data/cache/openmeteo_training"))
    else:
        required_weather = {
            "temp", "rh", "ws", "prec", "vapour_pressure_deficit", "soil_moisture_0_to_7cm"
        }
        if not required_weather.issubset(samples.columns):
            raise ValueError("--skip-weather için tüm hava ve toprak nemi öznitelikleri dosyada bulunmalı.")
        samples["weather_source"] = "Kullanıcı tarafından sağlanan saatlik/öncül hava verisi"

    samples = add_hotspot_persistence_features(samples, archive_events)
    before = len(samples)
    funnel["rows_before_feature_quality_filter"] = int(before)
    required_features_before_ffmc = [
        "landuse_code", "temp", "rh", "ws", "prec", "vapour_pressure_deficit",
        "soil_moisture_0_to_7cm", "doy", "hs_5km_1d", "hs_25km_3d", "label",
    ]
    samples = samples.dropna(subset=required_features_before_ffmc).copy()
    funnel["rows_removed_for_missing_required_features"] = int(before - len(samples))
    print(f"Eksik CORINE/konum/hava girdileri nedeniyle dışlanan kayıt: {before - len(samples)}")
    samples["landuse_code"] = pd.to_numeric(samples["landuse_code"], errors="coerce")
    samples["landuse_code"] = samples["landuse_code"].astype(int)
    samples = calculate_ffmc_features(samples, ffmc_seed=85.0)
    samples["target_name"] = "FIRMS thermal hotspot proxy; not confirmed illegal stubble burning"
    samples = add_spatial_groups(samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output_path, index=False)
    counts = samples["label"].value_counts().to_dict()
    funnel["final_positive_rows"] = int(counts.get(1, 0))
    funnel["final_negative_rows"] = int(counts.get(0, 0))
    funnel["final_spatial_groups_1deg"] = int(samples["spatial_group"].nunique())
    print("Veri kayıp/örnekleme hunisi:")
    for stage, value in funnel.items():
        print(f"  {stage}: {value:,}")
    metadata = {
        "dataset": str(output_path),
        "source_years": sorted(pd.to_datetime(samples["acq_date"]).dt.year.unique().astype(int).tolist()),
        "months": sorted({
            month
            for start, end in REGIONAL_HARVEST_WINDOWS.values()
            for month in range(start[0], end[0] + 1)
        }),
        "bbox_west_south_east_north": list(TURKEY_BBOX),
        "rows": int(len(samples)),
        "source_archive_event_rows_after_sensor_pixel_dedup": int(raw_archive_rows),
        "sampled_positive_event_rows_before_controls": int(len(candidate_hotspots)),
        "max_event_samples": max_event_samples,
        "positive_event_sampling_strata": "year × sensor × same 1-degree blocks used by spatial CV",
        "corine_lookup": corine_lookup_stats,
        "corine_lookup_outcomes": corine_outcomes if generate_controls else {},
        "corine_local_raster": str(corine_raster or DEFAULT_CORINE_RASTER) if (corine_raster or DEFAULT_CORINE_RASTER).exists() else None,
        "stage_funnel": funnel,
        "event_sample_seed": random_seed,
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "label_semantics": "firms_hotspot_proxy",
        "regional_harvest_calendar": {
            "windows": {
                REGIONS[code]: {
                    "start_mm_dd": f"{start[0]:02d}-{start[1]:02d}",
                    "end_mm_dd": f"{end[0]:02d}-{end[1]:02d}",
                }
                for code, (start, end) in REGIONAL_HARVEST_WINDOWS.items()
            },
            "region_rows": {
                str(region): int(count)
                for region, count in samples["harvest_region"].value_counts(dropna=False).items()
            },
            "region_month_rows": {
                str(region): {
                    str(int(month)): int(count)
                    for month, count in group.groupby(
                        pd.to_datetime(group["acq_date"]).dt.month
                    ).size().items()
                }
                for region, group in samples.groupby("harvest_region", dropna=False)
            },
            "source_years_with_training_rows": sorted(
                pd.to_datetime(samples["acq_date"]).dt.year.unique().astype(int).tolist()
            ),
            "archive_years_with_harvest_events": sorted(
                pd.to_datetime(regional_events["acq_date"]).dt.year.unique().astype(int).tolist()
            ),
            "source_links": HARVEST_SOURCES,
            "note": "Wheat/winter-cereal monitoring windows, not all crops. The ranges are broad and approximate; actual harvest shifts by crop, elevation, district and year. FIRMS labels remain thermal-hotspot proxies, not confirmed stubble fires.",
        },
        "positive_definition": "NASA FIRMS thermal hotspot detection; not legal status or independently confirmed fire cause",
        "negative_definition": (
            f"same {control_cell_size:g}-degree grid cell and exact CORINE agricultural class (2xx) as a positive; "
            "randomly selected from positive locations on dates without a same-cell FIRMS report; not field-verified fire-free"
            if generate_controls else
            "user-supplied field-verified-clear or no-FIRMS-detection control; not automatically true negative"
        ),
        "control_sampling": {
            "generated_from_archive_absence": generate_controls,
            "grid_cell_size_degrees": control_cell_size if generate_controls else None,
            "random_seed": random_seed if generate_controls else None,
            "sampling_unit": "grid_cell_day_random_positive_coordinate_exact_corine_match" if generate_controls else "user_control_point",
            "matched_corine_codes": sorted(matched_corine_codes) if generate_controls else [],
            "corine_profile": corine_profile,
            "expanded_profile_role": "sensitivity_analysis_only; do not present as the primary cereal stubble model",
        },
        "weather_spatial_sampling": {
            "max_era5_grid_locations": max_weather_grid_locations,
            "selected_grid_locations": len(selected_weather_cells) if selected_weather_cells is not None else None,
            "random_seed": random_seed,
            "note": "When capped, records outside deterministically sampled 0.25-degree ERA5 cells are excluded before feature retrieval.",
        },
        "weather": "Open-Meteo Historical Weather API ERA5 hourly; features from day before target at 12 UTC; 24-hour preceding precipitation sum; VPD and 0-7 cm soil moisture; 0.25-degree grid",
        "landcover": "Copernicus CORINE Land Cover 2018; broad class, not parcel-scale/current ground truth",
        "persistence_event_definition": {
            "sensors": list(PERSISTENCE_SENSOR_SET),
            "spatial_deduplication": f"fixed 39°N equirectangular {PERSISTENCE_GRID_SIZE_M:g} m grid × acquisition day",
            "note": "The same prepare_persistence_events function is called by training and live feature calculation.",
        },
        "ffmc_seed": 85.0,
        "ffmc_note": "CFFDRS daily FFMC state carried across consecutive ERA5 grid-day noon records; each discontinuous segment is explicitly re-initialized at 85.0, not a local observation",
        "predictors": MODEL_FEATURES,
        "harvest_features_in_model": HARVEST_MODEL_FEATURES,
        "excluded_leakage_fields": ["frp", "bright_ti4", "is_night"],
        "spatial_split_group": "1-degree blocks; all years from same block held in one fold",
    }
    output_path.with_suffix(".meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Veri seti yazıldı: {output_path} · {len(samples):,} satır · sınıflar={counts}")
    return samples


def main() -> None:
    # Keep diagnostics usable on Windows consoles that default to legacy code pages.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Türkiye 2020–2025 FIRMS/control eğitim veri seti oluşturucu")
    parser.add_argument("--firms-dir", type=Path, default=FIRMS_DIR)
    parser.add_argument("--controls", type=Path, default=CONTROLS_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--skip-corine", action="store_true", help="Yalnızca tüm FIRMS satırlarında kullanıcı tarafından landuse_code girilmişse kullanın.")
    parser.add_argument("--skip-weather", action="store_true", help="Yalnızca tüm satırlarda geçerli temp/rh/ws/prec varsa kullanın.")
    parser.add_argument("--generate-no-detection-controls", action="store_true", help="Arşivde aynı hücre/ay için tespit olmayan günleri vekil negatif kontrol olarak örnekler.")
    parser.add_argument("--control-cell-size", type=float, default=0.1, help="Örnekleme hücresi derece (varsayılan 0.1, yaklaşık 11 km).")
    parser.add_argument("--max-weather-grid-locations", type=int, default=1000, help="Gerçek ERA5 verisi alınacak 0.25° hücre sayısı üst sınırı; tüm konumlar için 0 girilemez. Varsayılan 1000.")
    parser.add_argument("--max-event-samples", type=int, default=20_000, help="Yıl/sensör/2° bölge tabakalı pozitif olay üst sınırı (varsayılan 20.000). 0 verilirse tüm olaylar kullanılır.")
    parser.add_argument("--corine-raster", type=Path, default=DEFAULT_CORINE_RASTER, help="İsteğe bağlı Copernicus CLC 2018 100 m GeoTIFF; varsa uzak nokta sorgusu yerine rasterio ile örneklenir.")
    parser.add_argument("--corine-profile", choices=sorted(CORINE_PROFILES), default="grain", help="Ana model için tahıl/arable sınıfları (grain); genişletilmiş tüm tarım sınıfları yalnız duyarlılık analizi içindir.")
    args = parser.parse_args()
    build_dataset(
        args.firms_dir, args.controls, args.output,
        skip_corine=args.skip_corine, skip_weather=args.skip_weather,
        generate_controls=args.generate_no_detection_controls,
        control_cell_size=args.control_cell_size,
        max_weather_grid_locations=args.max_weather_grid_locations,
        max_event_samples=None if args.max_event_samples == 0 else args.max_event_samples,
        corine_raster=args.corine_raster if args.corine_raster.exists() else None,
        corine_profile=args.corine_profile,
    )


if __name__ == "__main__":
    main()
