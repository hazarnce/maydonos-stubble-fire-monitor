"""Validated, leakage-aware features for the MaydonOs Turkey fire prototype."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from ml.harvest_calendar import HARVEST_MODEL_FEATURES

try:
    from cffdrs.fwi import fine_fuel_moisture_code
except ImportError:  # Report a clear error when the training extra is not installed.
    fine_fuel_moisture_code = None

# FRP, brightness and is_night are retained in the data table for audit/reporting.
# They are not predictors: FRP/brightness exist only after a hotspot is detected,
# and is_night is commonly absent for controls. Using them would leak the target.
MODEL_FEATURES = [
    "temp", "rh", "ws", "prec", "ffmc", "landuse_code", "doy",
    "hs_5km_1d", "hs_25km_3d", "vapour_pressure_deficit",
    "soil_moisture_0_to_7cm", *HARVEST_MODEL_FEATURES,
]
PERSISTENCE_FEATURES = ["hs_5km_1d", "hs_25km_3d"]
PERSISTENCE_SENSOR_SET = ("SNPP", "N20")
PERSISTENCE_GRID_SIZE_M = 375.0
AUDIT_ONLY_FEATURES = ["frp", "bright_ti4", "is_night", "landuse_label", "label_basis"]
WEATHER_INPUTS = ["temp", "rh", "ws", "prec"]
LANDUSE_CODES = {"other": 0, "farmland": 1, "forest": 2, "industrial": 3}


def normalize_firms_sensor(row: pd.Series | dict) -> str | None:
    """Map archive/API sensor labels to the two sensors used in both pipelines."""
    source = str(row.get("firms_source", "")).strip().upper()
    satellite = str(row.get("satellite", "")).strip().upper().replace("_", "-")
    if "SNPP" in source or "SUOMI" in source or satellite in {"SNPP", "N", "NPP", "SUOMI-NPP"}:
        return "SNPP"
    if "NOAA20" in source or "NOAA-20" in source or "J1V" in source or satellite in {"N20", "NOAA-20", "NOAA20", "J1"}:
        return "N20"
    if "NOAA21" in source or "NOAA-21" in source or satellite in {"N21", "NOAA-21", "NOAA21", "J2"}:
        return "N21"
    if "MODIS" in source or satellite in {"A", "T", "AQUA", "TERRA", "MODIS"}:
        return "MODIS"
    return None


def normalize_firms_sensors(frame: pd.DataFrame) -> pd.Series:
    """Vectorized counterpart of :func:`normalize_firms_sensor` for archives."""
    source = frame.get("firms_source", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip().str.upper()
    satellite = frame.get("satellite", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip().str.upper().str.replace("_", "-", regex=False)
    result = pd.Series(pd.NA, index=frame.index, dtype="object")
    snpp = source.str.contains("SNPP|SUOMI", regex=True) | satellite.isin({"SNPP", "N", "NPP", "SUOMI-NPP"})
    n20 = source.str.contains("NOAA20|NOAA-20|J1V", regex=True) | satellite.isin({"N20", "NOAA-20", "NOAA20", "J1"})
    n21 = source.str.contains("NOAA21|NOAA-21", regex=True) | satellite.isin({"N21", "NOAA-21", "NOAA21", "J2"})
    modis = source.str.contains("MODIS", regex=False) | satellite.isin({"A", "T", "AQUA", "TERRA", "MODIS"})
    result.loc[modis] = "MODIS"
    result.loc[n21] = "N21"
    result.loc[n20] = "N20"
    result.loc[snpp] = "SNPP"
    return result


def prepare_persistence_events(events: pd.DataFrame) -> pd.DataFrame:
    """Use the same S-NPP/NOAA-20 375 m × acquisition-day event set everywhere.

    A fixed equirectangular grid at 39°N gives a consistent, auditable spatial
    key over Turkey. This is a pixel-scale de-duplication proxy, not a perfect
    sensor geolocation correction.
    """
    required = {"latitude", "longitude", "acq_date"}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError("Persistence olaylarında eksik alan: " + ", ".join(sorted(missing)))
    if "satellite" not in events and "firms_source" not in events:
        raise ValueError("Persistence için uydu kaynağı belirtilmeli; sensörsüz sayım yapılmadı.")
    out = events.copy()
    if "is_demo" in out:
        out = out.loc[~out["is_demo"].fillna(False).astype(bool)].copy()
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out["_date"] = pd.to_datetime(out["acq_date"], errors="coerce").dt.normalize()
    out["_sensor"] = normalize_firms_sensors(out)
    out = out.loc[out["_sensor"].isin(PERSISTENCE_SENSOR_SET)].copy()
    if out.empty:
        return out
    out = out.dropna(subset=["latitude", "longitude", "_date"])
    out = out.loc[
        out["latitude"].between(-90, 90) & out["longitude"].between(-180, 180)
    ].copy()
    if out.empty:
        return out
    cell = PERSISTENCE_GRID_SIZE_M
    out["_grid_y"] = np.floor(out["latitude"] * 111_320.0 / cell).astype("int64")
    out["_grid_x"] = np.floor(
        out["longitude"] * 111_320.0 * math.cos(math.radians(39.0)) / cell
    ).astype("int64")
    out["_frp_sort"] = pd.to_numeric(out.get("frp", np.nan), errors="coerce")
    out = out.sort_values(
        ["_date", "_grid_y", "_grid_x", "_frp_sort", "_sensor"],
        ascending=[True, True, True, False, True], na_position="last",
    )
    out = out.drop_duplicates(["_date", "_grid_y", "_grid_x"], keep="first")
    return out.drop(columns=["_grid_y", "_grid_x", "_frp_sort", "_date", "_sensor"]).reset_index(drop=True)


def add_hotspot_persistence_features(
    samples: pd.DataFrame, hotspot_events: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Add strictly backward-looking FIRMS counts; same-day events are excluded.

    Distances are great-circle radii. Missing coordinates/dates are rejected; no
    persistence value is silently substituted for an unavailable target day.
    """
    required = {"latitude", "longitude", "acq_date"}
    if not required.issubset(samples.columns):
        raise ValueError("Persistence için örneklerde latitude/longitude/acq_date gerekli.")
    raw_events = hotspot_events if hotspot_events is not None else samples
    if not required.issubset(raw_events.columns):
        raise ValueError("Persistence olaylarında latitude/longitude/acq_date gerekli.")

    out = samples.copy()
    event_data = prepare_persistence_events(raw_events)[["latitude", "longitude", "acq_date"]].copy()
    event_data["_date"] = pd.to_datetime(event_data["acq_date"], errors="coerce").dt.normalize()
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out["_date"] = pd.to_datetime(out["acq_date"], errors="coerce").dt.normalize()
    if out[["latitude", "longitude", "_date"]].isna().any().any():
        raise ValueError("Persistence örneklerinde eksik/bozuk koordinat veya tarih var.")
    event_data = event_data.dropna(subset=["latitude", "longitude", "_date"])
    if event_data.empty:
        raise ValueError("Persistence için S-NPP/NOAA-20 FIRMS olay kaydı yok.")

    out["doy"] = out["_date"].dt.dayofyear.astype(int)
    out["hs_5km_1d"] = 0
    out["hs_25km_3d"] = 0
    from sklearn.neighbors import BallTree

    earth_radius_km = 6371.0088
    for day, indices in out.groupby("_date", sort=False).groups.items():
        day = pd.Timestamp(day)
        query_rows = out.loc[indices, ["latitude", "longitude"]].to_numpy(dtype=float)
        windows = (
            ("hs_5km_1d", day - pd.Timedelta(days=1), 5.0),
            ("hs_25km_3d", day - pd.Timedelta(days=3), 25.0),
        )
        for column, start, radius_km in windows:
            prior = event_data.loc[
                event_data["_date"].ge(start) & event_data["_date"].lt(day),
                ["latitude", "longitude"],
            ]
            if prior.empty:
                continue
            tree = BallTree(np.radians(prior.to_numpy(dtype=float)), metric="haversine")
            counts = tree.query_radius(
                np.radians(query_rows), r=radius_km / earth_radius_km, count_only=True
            )
            out.loc[indices, column] = counts.astype(int)
    return out.drop(columns=["_date"])


def _finite_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column not in out.columns:
            raise ValueError(f"Gerekli sütun eksik: {column}")
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def calculate_ffmc_features(
    frame: pd.DataFrame,
    *,
    ffmc_seed: float = 85.0,
    previous_ffmc_column: str = "ffmc_yesterday",
) -> pd.DataFrame:
    """Compute the documented daily CFFDRS FFMC step for every sample.

    `prec` must be the complete 24-hour precipitation amount for the preceding
    weather day. If `ffmc_yesterday` is not supplied, `ffmc_seed` is an explicit
    initialization value, not a field observation.
    """
    if fine_fuel_moisture_code is None:
        raise RuntimeError("CFFDRS kurulu değil. `pip install -r requirements-training.txt` çalıştırın.")
    out = _finite_numeric(frame, [*WEATHER_INPUTS])
    if "tmax" in out.columns and "tmin" in out.columns:
        highs = pd.to_numeric(out["tmax"], errors="coerce")
        lows = pd.to_numeric(out["tmin"], errors="coerce")
        bad = highs.notna() & lows.notna() & (highs < lows)
        if bad.any():
            raise ValueError(f"Tmax < Tmin olan {int(bad.sum())} satır var.")
    invalid = (
        ~np.isfinite(out[WEATHER_INPUTS]).all(axis=1)
        | (out["temp"] < -60) | (out["temp"] > 60)
        | (out["rh"] < 0) | (out["rh"] > 100)
        | (out["ws"] < 0) | (out["prec"] < 0)
    )
    if invalid.any():
        raise ValueError(f"Geçersiz/eksik hava verisi içeren {int(invalid.sum())} satır var; veri doldurulmadı.")

    if previous_ffmc_column in out.columns:
        previous = pd.to_numeric(out[previous_ffmc_column], errors="coerce")
        if previous.isna().any() or not np.isfinite(previous).all() or ((previous < 0) | (previous > 101)).any():
            raise ValueError("ffmc_yesterday değerleri 0–101 aralığında ve eksiksiz olmalı.")
        seeds = previous.to_numpy(dtype=float)
        out["ffmc_seed_basis"] = "satırdaki_ffmc_yesterday"
    else:
        if not math.isfinite(float(ffmc_seed)) or not 0 <= float(ffmc_seed) <= 101:
            raise ValueError("FFMC başlangıç değeri 0–101 aralığında olmalı.")
        seeds = np.full(len(out), float(ffmc_seed), dtype=float)
        out["ffmc_seed_basis"] = f"açık_başlangıç_değeri_{float(ffmc_seed):g}"

    ffmc_values: list[float] = []
    for seed, row in zip(seeds, out.itertuples(index=False)):
        values = [float(getattr(row, name)) for name in WEATHER_INPUTS]
        try:
            value = float(fine_fuel_moisture_code(
                ffmc_yda=float(seed), temp=values[0], rh=values[1], ws=values[2], prec=values[3]
            ))
        except Exception as exc:
            raise ValueError(f"CFFDRS FFMC hesaplanamadı: {exc}") from exc
        if not math.isfinite(value):
            raise ValueError("CFFDRS geçersiz FFMC üretti.")
        ffmc_values.append(float(np.clip(value, 0.0, 99.0)))
    out["ffmc"] = ffmc_values
    return out


def prepare_model_table(
    frame: pd.DataFrame, features: Iterable[str] = MODEL_FEATURES
) -> tuple[pd.DataFrame, float]:
    """Validate labels/features and calculate the positive class weight.

    FRP, brightness and night/day are intentionally not placed in X. They remain
    in the returned audit table so researchers can inspect event attributes.
    """
    out = frame.copy()
    selected_features = list(features)
    required = [*selected_features, "label"]
    missing = [name for name in required if name not in out]
    if missing:
        raise ValueError("Eğitim verisinde eksik sütunlar: " + ", ".join(missing))
    for name in [*selected_features, "label"]:
        out[name] = pd.to_numeric(out[name], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    invalid = out[[*selected_features, "label"]].isna().any(axis=1)
    if invalid.any():
        raise ValueError(f"Model girdilerinde eksik/geçersiz {int(invalid.sum())} satır var; imputasyon yapılmadı.")
    if not out["label"].isin([0, 1]).all():
        raise ValueError("label yalnızca 0 veya 1 olabilir.")
    if "landuse_code" in selected_features and not out["landuse_code"].isin([0, 1, 2, 3]).all():
        raise ValueError("landuse_code yalnızca 0=diğer, 1=tarım, 2=orman, 3=sanayi olabilir.")
    counts = out["label"].value_counts()
    if set(counts.index.astype(int)) != {0, 1} or counts.get(1, 0) == 0:
        raise ValueError("Eğitim için pozitif ve negatif örneklerin ikisi de bulunmalı.")
    weight = float(counts[0] / counts[1])
    return out, weight
