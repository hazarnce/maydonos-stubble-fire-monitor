"""Guarded inference for the exploratory FIRMS hotspot ranking model."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from backend.engine import calculate_fire_risk
from ml.feature_engineering import (
    PERSISTENCE_FEATURES,
    PERSISTENCE_SENSOR_SET,
    add_hotspot_persistence_features,
    normalize_firms_sensor,
    normalize_firms_sensors,
)
from ml.harvest_calendar import (
    HARVEST_MODEL_FEATURES,
    REGIONS,
    harvest_model_features,
    is_in_harvest_window,
    province_at,
    region_code_for_province,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "ml" / "models" / "xgboost_turkey_stubble_model.json"
MINIMUM_DISPLAY_AUC = 0.65


@lru_cache(maxsize=1)
def _load_model(model_mtime: int, metrics_mtime: int):
    from xgboost import XGBClassifier

    metrics_path = MODEL_PATH.with_suffix(".metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    features = tuple(metrics.get("features") or ())
    if not features:
        raise ValueError("Model öznitelikleri tanımlı değil.")
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    names = tuple(model.get_booster().feature_names or ())
    if names and names != features:
        raise ValueError("Model ve metrik öznitelikleri uyuşmuyor.")
    return model, features, metrics


def _corine_model_code(value) -> int | None:
    try:
        code = int(float(str(value).strip()))
    except (ValueError, TypeError, OverflowError):
        return None
    if 200 <= code < 300:
        return 1
    if 300 <= code < 400:
        return 2
    if code in {121, 131, 132, 133}:
        return 3
    return 0


def _daily_weather(weather: pd.DataFrame | None) -> dict:
    if weather is None or weather.empty or "time" not in weather:
        return {}
    frame = weather.copy().sort_values("time")
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time"])
    frame["precipitation"] = pd.to_numeric(frame["precipitation"], errors="coerce")
    snapshots = frame.loc[frame.time.dt.hour.eq(15)]
    results: dict = {}
    seed = 85.0
    previous_day = None
    for row in snapshots.itertuples(index=False):
        stamp = pd.Timestamp(row.time)
        day = stamp.normalize()
        if previous_day is None or (day - previous_day).days != 1:
            seed = 85.0
        window = frame.loc[(frame.time > stamp - pd.Timedelta(hours=24)) & (frame.time <= stamp), "precipitation"]
        if len(window) != 24 or window.isna().any() or (window < 0).any():
            seed = 85.0
            previous_day = day
            continue
        try:
            temp = float(row.temperature_2m)
            rh = float(row.relative_humidity_2m)
            ws = float(row.wind_speed_10m)
            rain = float(window.sum())
            ffmc = calculate_fire_risk(temp, rh, ws, rain, seed)
        except (ValueError, TypeError, KeyError, OverflowError, RuntimeError):
            seed = 85.0
            previous_day = day
            continue
        values = {"temp": temp, "rh": rh, "ws": ws, "prec": rain, "ffmc": ffmc}
        for name in ("vapour_pressure_deficit", "soil_moisture_0_to_7cm"):
            value = pd.to_numeric(getattr(row, name, np.nan), errors="coerce")
            if pd.notna(value) and np.isfinite(float(value)):
                values[name] = float(value)
        results[day.date()] = values
        seed, previous_day = ffmc, day
    return results


def score_live_events(
    events: pd.DataFrame,
    display_rows: pd.DataFrame,
    weather: pd.DataFrame | None,
) -> tuple[pd.DataFrame, str, dict | None]:
    """Rank eligible district events; never return the proxy score as probability."""
    output = display_rows.copy()
    output["model_rank_percentile"] = np.nan
    if output.empty:
        return output, "Görüntülenecek nokta yok.", None
    if weather is None:
        return output, "Model sıralaması gizli: ülke/il görünümünde tek hava noktasını tüm olaylara atamak yanıltıcı olur.", None
    if not MODEL_PATH.is_file() or not MODEL_PATH.with_suffix(".metrics.json").is_file():
        return output, "Model sıralaması gizli: model veya doğrulama metrik dosyası yok.", None
    try:
        metrics_path = MODEL_PATH.with_suffix(".metrics.json")
        model, features, metrics = _load_model(MODEL_PATH.stat().st_mtime_ns, metrics_path.stat().st_mtime_ns)
    except Exception as exc:
        return output, f"Model sıralaması gizli: {type(exc).__name__}.", None
    if metrics.get("label_semantics") != "firms_hotspot_proxy":
        return output, "Model etiketi desteklenmiyor; sıralama üretilmedi.", None
    if not set(HARVEST_MODEL_FEATURES).issubset(features):
        return output, "Model sıralaması gizli: model bölgesel hasat dönemi girdileriyle yeniden eğitilmelidir.", None
    auc = pd.to_numeric(metrics.get("spatial_cv_auc_mean"), errors="coerce")
    if pd.isna(auc) or float(auc) < MINIMUM_DISPLAY_AUC:
        return output, "Model sıralaması gizli: dört katlı coğrafi AUC gösterim eşiğinin altında.", None
    daily = _daily_weather(weather)
    if not daily:
        return output, "Model sıralaması gizli: olaydan önceki güne ait eksiksiz hava/yağış kaydı yok.", None
    persistence = {"hs_5km_1d", "hs_25km_3d"}.intersection(features)
    if persistence and persistence != set(PERSISTENCE_FEATURES):
        return output, "Model sıralaması gizli: persistence özellik sözleşmesi uyuşmuyor.", None
    if persistence:
        eligible = output.loc[normalize_firms_sensors(output).isin(PERSISTENCE_SENSOR_SET)].copy()
        if eligible.empty:
            return output, "Model sıralaması gizli: S-NPP/NOAA-20 olay kaydı yok.", None
        try:
            persistence_rows = add_hotspot_persistence_features(eligible, events)
            persistence_by_index = persistence_rows[PERSISTENCE_FEATURES].to_dict(orient="index")
        except Exception as exc:
            return output, f"Model sıralaması gizli: önceki olay verisi eksik ({type(exc).__name__}).", None
    else:
        persistence_by_index = {}

    samples: list[dict[str, float]] = []
    indices: list[object] = []
    season_counts = {code: 0 for code in REGIONS}
    outside_harvest_window = 0
    unknown_region = 0
    sensor_event_count = 0
    for index, row in output.iterrows():
        if normalize_firms_sensor(row) not in PERSISTENCE_SENSOR_SET:
            continue
        sensor_event_count += 1
        try:
            event_day = pd.Timestamp(row.acq_date).date()
            province = row.get("province") or row.get("province_name")
            if not province:
                province = province_at(row.get("latitude"), row.get("longitude"))
            region_code = region_code_for_province(province)
            if region_code is None:
                unknown_region += 1
                continue
            if not is_in_harvest_window(event_day, region_code):
                outside_harvest_window += 1
                continue
            weather_values = daily.get((pd.Timestamp(event_day) - pd.Timedelta(days=1)).date())
            landuse = _corine_model_code(row.get("corine_code"))
            if weather_values is None or landuse is None:
                continue
            values = {**weather_values, "landuse_code": float(landuse), "doy": float(pd.Timestamp(event_day).dayofyear)}
            values.update(harvest_model_features(event_day, region_code))
            if persistence:
                prior = persistence_by_index.get(index)
                if prior is None:
                    continue
                values.update({key: float(prior[key]) for key in PERSISTENCE_FEATURES})
            if any(name not in values or not np.isfinite(values[name]) for name in features):
                continue
            samples.append({name: float(values[name]) for name in features})
            indices.append(index)
            season_counts[region_code] += 1
        except (ValueError, TypeError, KeyError, OverflowError):
            continue
    if len(samples) < 2:
        if sensor_event_count and outside_harvest_window == sensor_event_count and unknown_region == 0:
            message = "Model sıralaması üretilmedi: noktalar bulundukları ilin yaklaşık hububat hasat penceresinin dışında."
        elif unknown_region:
            message = "Model sıralaması gizli: bazı noktalar için il/bölge hasat takvimi eşleşmedi."
        else:
            message = "Model sıralaması gizli: en az iki noktada aynı hava, CORINE, uydu ve geçmiş girdileri eşleşmedi."
        return output, message, {
            "auc_mean": float(auc), "auc_range": metrics.get("spatial_cv_auc_range"),
            "eligible": len(samples), "outside_harvest_window": outside_harvest_window,
            "unknown_region": unknown_region,
        }
    try:
        raw_scores = model.predict_proba(pd.DataFrame(samples, columns=features))[:, 1]
    except Exception as exc:
        return output, f"Model sıralaması gizli: tahmin girdisi reddedildi ({type(exc).__name__}).", None
    ranks = pd.Series(raw_scores).rank(method="average", pct=True).to_numpy() * 100.0
    for index, rank in zip(indices, ranks):
        output.at[index, "model_rank_percentile"] = float(rank)
    info = {
        "auc_mean": float(auc),
        "auc_range": metrics.get("spatial_cv_auc_range"),
        "eligible": len(samples),
        "scored_rows": len(indices),
        "harvest_regions_scored": {
            REGIONS[code]: count for code, count in season_counts.items() if count
        },
        "outside_harvest_window": outside_harvest_window,
        "evaluation_rows": metrics.get("test_rows_total"),
        "label_semantics": "firms_hotspot_proxy",
    }
    return output, "Göreli sıra: yalnız bu görünümdeki uygun FIRMS noktaları arasında. Olasılık veya saha doğrulaması değildir.", info
