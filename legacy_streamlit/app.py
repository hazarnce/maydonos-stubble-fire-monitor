"""MaydonOs • Diyarbakır anız ve yangın hava koşulları erken uyarı prototipi."""

from __future__ import annotations

try:
    import truststore

    truststore.inject_into_ssl()
    SYSTEM_CA_STORE_ENABLED = True
except (ImportError, AttributeError):
    SYSTEM_CA_STORE_ENABLED = False

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
import sys

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.engine import (
    anomaly_priority,
    bbox_from_center,
    calculate_cffdrs_system,
    calculate_fire_risk,
    calculate_hourly_ffmc_series,
    classify_ffmc,
    clear_firms_hotspots_cache,
    fetch_nasa_firms_hotspots,
    fetch_open_meteo_weather,
    previous_24h_rain,
    secret_or_environment,
    validate_coordinates,
)
from backend.hotspot_anomaly import record_snapshot, score_snapshot
from backend.thermal_context import enrich_hotspots_with_osm
from backend.corine_context import classify_hotspot_tr, enrich_hotspots_with_corine
from backend.vision_analysis import (
    LAND_COVER_LABELS,
    SURFACE_STATE_LABELS,
    analyze_overhead_scene,
    esri_export_url,
    fetch_esri_snapshot,
)
from ml.feature_engineering import (
    PERSISTENCE_FEATURES,
    PERSISTENCE_SENSOR_SET,
    add_hotspot_persistence_features,
    normalize_firms_sensor,
    normalize_firms_sensors,
)

MIN_MODEL_DISPLAY_AUC = 0.65


DISTRICTS: dict[str, tuple[float, float, float]] = {
    "Tüm Kent": (37.9100, 40.2300, 0.55),
    "Diyarbakır Merkez": (37.9100, 40.2300, 0.18),
    "Bismil": (37.8478, 40.6628, 0.20),
    "Çınar": (37.7253, 40.4156, 0.20),
    "Sur": (37.9144, 40.2386, 0.12),
    "Kayapınar": (37.9400, 40.1800, 0.12),
    "Ergani": (38.2667, 39.7597, 0.20),
}

st.set_page_config(
    page_title="MaydonOs | Anız erken uyarı",
    page_icon=":material/local_fire_department:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: max(1.4rem, env(safe-area-inset-top));
        padding-bottom: max(2rem, env(safe-area-inset-bottom));
        max-width: 1480px;
    }
    button, [role="button"], input, select, textarea { min-height: 44px; }
    button:focus-visible, a:focus-visible, input:focus-visible,
    [tabindex]:focus-visible { outline: 3px solid #d59a43 !important; outline-offset: 3px; }
    [data-testid="stAlert"] { border-radius: 14px; }
    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
        border-radius: 16px;
        padding: 14px 16px;
        min-height: 106px;
    }
    [data-testid="stMetricLabel"] p { font-size: .82rem; line-height: 1.35; }
    [data-testid="stTabs"] [role="tab"] { min-height: 48px; }
    [data-testid="stDataFrame"], [data-testid="stTable"] { border-radius: 14px; overflow: hidden; }
    .maydonos-masthead {
        background: linear-gradient(118deg, #410002 0%, #650019 68%, #850122 100%);
        color: #fffaf3; border-radius: 22px;
        padding: clamp(22px, 4vw, 38px); margin: 2px 0 22px;
        border-bottom: 5px solid #d59a43;
        box-shadow: 0 12px 32px rgba(65, 0, 2, .12);
    }
    .maydonos-masthead .brand-row { display:flex; align-items:center; gap:12px; margin-bottom:20px; }
    .maydonos-masthead .brand-mark {
        display:grid; place-items:center; width:42px; height:42px; border-radius:14px;
        background:#fffaf3; color:#850122; font-weight:800; font-size:1.15rem;
    }
    .maydonos-masthead .brand-name { color:#fffaf3; font-weight:750; letter-spacing:.01em; }
    .maydonos-masthead .brand-area {
        margin-left:auto; border:1px solid rgba(255,250,243,.35); border-radius:999px;
        color:#fffaf3; padding:7px 12px; font-size:.82rem;
    }
    .maydonos-masthead .brand-kicker { color:#f2c4bd; font-size:.78rem;
        letter-spacing:.075em; text-transform:uppercase; margin:0 0 9px; }
    .maydonos-masthead h1 { color:#fffaf3 !important; margin:0 0 9px;
        font-size:clamp(1.8rem, 4vw, 2.7rem); line-height:1.1; letter-spacing:-.025em; }
    .maydonos-masthead p { color:#f8e9e2; margin:0; max-width:68ch; font-size:1.02rem; line-height:1.55; }
    .section-intro { color:var(--text-color); opacity:.78; margin-top:-.55rem; margin-bottom:1rem; }
    .risk-summary {
        display:flex; align-items:center; gap:14px; margin:14px 0 8px; padding:16px 18px;
        color:var(--text-color); background:var(--secondary-background-color);
        border:1px solid color-mix(in srgb, var(--text-color) 13%, transparent);
        border-left:5px solid var(--risk-accent); border-radius:15px;
    }
    .risk-summary strong { font-size:1.04rem; }
    .risk-dot { flex:0 0 13px; width:13px; height:13px; border-radius:50%; background:var(--risk-accent); }
    .risk-detail { margin-top:4px; opacity:.78; font-size:.9rem; line-height:1.45; }
    @media (max-width: 640px) {
        .block-container { padding-left: max(1rem, env(safe-area-inset-left));
            padding-right: max(1rem, env(safe-area-inset-right)); }
        .maydonos-masthead { border-radius:17px; }
        .maydonos-masthead .brand-area { display:none; }
        [data-testid="stMetric"] { min-height:92px; padding:11px 12px; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { scroll-behavior: auto !important;
            animation-duration: .01ms !important; animation-iteration-count: 1 !important;
            transition-duration: .01ms !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_dashboard_css = PROJECT_ROOT / "legacy_streamlit" / "static" / "dashboard.css"
if _dashboard_css.is_file():
    st.markdown(
        f"<style>{_dashboard_css.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=60 * 60 * 24 * 30, show_spinner=False)
def reverse_geocode(lat: float, lon: float) -> str | None:
    """Resolve a selected FIRMS coordinate to a nearby address."""
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 16, "accept-language": "tr"},
            headers={"User-Agent": "MaydonOs-Prototype/1.0 (Diyarbakir fire monitoring)"},
            timeout=(4, 8),
        )
        response.raise_for_status()
        address = response.json().get("display_name")
        return str(address) if address else None
    except (requests.RequestException, ValueError, TypeError):
        return None


@st.cache_resource(show_spinner=False)
def _load_hotspot_model(model_path: str, model_mtime_ns: int, metrics_mtime_ns: int):
    """Load the saved XGBoost model and its declared feature contract once."""
    model_file = Path(model_path)
    metrics_path = model_file.with_suffix(".metrics.json")
    try:
        import json
        from xgboost import XGBClassifier

        metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
        features = tuple(metadata.get("features") or ())
        if not features:
            raise ValueError("Modelin beklediği öznitelikler metrik dosyasında tanımlı değil.")
        model = XGBClassifier()
        model.load_model(str(model_file))
        booster_features = tuple(model.get_booster().feature_names or ())
        if booster_features and booster_features != features:
            raise ValueError("Model dosyasındaki öznitelikler ile metrik dosyasındaki liste uyuşmuyor.")
        return model, features, metadata.get("label_semantics"), metadata, None
    except Exception as exc:
        return None, (), None, {}, f"Model yüklenemedi: {type(exc).__name__}: {exc}"


def load_hotspot_model():
    """Reload automatically when either model or metrics changes on disk."""
    model_path = PROJECT_ROOT / "ml" / "models" / "xgboost_turkey_stubble_model.json"
    metrics_path = model_path.with_suffix(".metrics.json")
    if not model_path.is_file() or not metrics_path.is_file():
        return None, (), None, {}, "Eğitilmiş model dosyası veya metrik bildirimi bulunamadı."
    return _load_hotspot_model(
        str(model_path), model_path.stat().st_mtime_ns, metrics_path.stat().st_mtime_ns
    )


def _corine_model_code(value) -> int | None:
    if value is None or pd.isna(value):
        return None
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


def _daily_model_weather(weather: pd.DataFrame | None) -> dict[object, dict[str, float]]:
    """Build event-date−1 noon features from the local hourly weather response."""
    if weather is None or weather.empty or "time" not in weather:
        return {}
    frame = weather.sort_values("time").copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["time"])
    frame["precipitation"] = pd.to_numeric(frame["precipitation"], errors="coerce")
    frame["_day"] = frame["time"].dt.normalize()
    frame["_hour"] = frame["time"].dt.hour
    snapshots = frame.loc[frame["_hour"].eq(15)].copy()  # 12:00 UTC in Diyarbakır local time
    result: dict[object, dict[str, float]] = {}
    ffmc_seed = 85.0  # same explicit initialization used by the training pipeline
    previous_day = None
    for row in snapshots.itertuples(index=False):
        stamp = pd.Timestamp(getattr(row, "time"))
        day = stamp.normalize()
        if previous_day is None or (day - previous_day).days != 1:
            ffmc_seed = 85.0
        start = stamp - pd.Timedelta(hours=24)
        rain_window = frame.loc[frame["time"].gt(start) & frame["time"].le(stamp), "precipitation"]
        if len(rain_window) != 24 or rain_window.isna().any() or (rain_window < 0).any():
            previous_day = day
            continue
        source = frame.loc[frame["time"].eq(stamp)]
        if source.empty:
            previous_day = day
            continue
        source_row = source.iloc[-1]
        try:
            temperature = float(source_row["temperature_2m"])
            humidity = float(source_row["relative_humidity_2m"])
            wind = float(source_row["wind_speed_10m"])
            rain = float(rain_window.sum())
            ffmc_value = calculate_fire_risk(temperature, humidity, wind, rain, ffmc_seed)
        except (RuntimeError, ValueError, TypeError, KeyError, OverflowError):
            previous_day = day
            continue
        values = {
            "temp": temperature, "rh": humidity, "ws": wind, "prec": rain,
            "ffmc": ffmc_value,
        }
        for name in ("vapour_pressure_deficit", "soil_moisture_0_to_7cm"):
            value = pd.to_numeric(source_row.get(name), errors="coerce")
            if pd.notna(value) and np.isfinite(float(value)):
                values[name] = float(value)
        result[day.date()] = values
        ffmc_seed = ffmc_value
        previous_day = day
    return result


def score_hotspots_with_xgboost(
    hotspots: pd.DataFrame,
    weather: pd.DataFrame | None,
    *,
    allow_persistence: bool = True,
    persistence_events: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, str]:
    """Add a within-view proxy-model rank when every required input is available."""
    result = hotspots.copy()
    result["xgboost_rank_percentile"] = np.nan
    model, features, semantics, metrics, error = load_hotspot_model()
    if error:
        return result, error
    if result.empty:
        return result, "Görüntülenecek FIRMS noktası yok."
    if semantics != "firms_hotspot_proxy":
        return result, "Model etiketi bu arayüzde desteklenmiyor; puan üretilmedi."
    cv_auc = pd.to_numeric(metrics.get("spatial_cv_auc_mean"), errors="coerce")
    if pd.isna(cv_auc) or float(cv_auc) < MIN_MODEL_DISPLAY_AUC:
        if pd.isna(cv_auc):
            return result, "Model sıralaması gizli: yeni dört katlı coğrafi doğrulama sonucu bulunmuyor."
        return result, "Model sıralaması gizli: dört katlı coğrafi doğrulama gösterim eşiğini karşılamıyor."
    daily_weather = _daily_model_weather(weather)
    if not daily_weather:
        return result, "Model için olay öncesi öğle hava kaydı veya eksiksiz 24 saat yağış bulunamadı."
    required_persistence = {"hs_5km_1d", "hs_25km_3d"}.intersection(features)
    if required_persistence and required_persistence != set(PERSISTENCE_FEATURES):
        return result, "Model persistence öznitelik tanımı beklenenle uyuşmuyor; puan üretilmedi."
    if required_persistence and not allow_persistence:
        return result, "FIRMS sensör kapsamı eksik; komşu olay öznitelikleri hesaplanmadı."
    persistence_by_index: dict[object, dict[str, float]] = {}
    if required_persistence:
        eligible = result.loc[
            ~result.get("is_demo", pd.Series(False, index=result.index)).fillna(False).astype(bool)
            & normalize_firms_sensors(result).isin(PERSISTENCE_SENSOR_SET)
        ].copy()
        try:
            if eligible.empty:
                raise ValueError("Model için eşleşen S-NPP/NOAA-20 noktası yok.")
            event_source = persistence_events if persistence_events is not None else hotspots
            with_persistence = add_hotspot_persistence_features(eligible, event_source)
            persistence_by_index = with_persistence[PERSISTENCE_FEATURES].to_dict(orient="index")
        except (ValueError, TypeError, KeyError) as exc:
            return result, f"Ortak persistence hesabı kullanılamadı; puan üretilmedi ({exc})."
    inputs: list[dict[str, float]] = []
    row_indices: list[object] = []
    for idx, row in result.iterrows():
        if bool(row.get("is_demo", False)):
            continue
        if normalize_firms_sensor(row) not in PERSISTENCE_SENSOR_SET:
            continue
        try:
            event_day = pd.Timestamp(row["acq_date"]).date()
            weather_day = (pd.Timestamp(event_day) - pd.Timedelta(days=1)).date()
            weather_values = daily_weather.get(weather_day)
            landuse_code = _corine_model_code(row.get("corine_code"))
            if weather_values is None or landuse_code is None:
                continue
            values = {
                **weather_values,
                "landuse_code": float(landuse_code),
                "doy": float(pd.Timestamp(event_day).dayofyear),
            }
            if required_persistence:
                persistence = persistence_by_index.get(idx)
                if persistence is None:
                    continue
                values.update({name: float(persistence[name]) for name in PERSISTENCE_FEATURES})
            if any(name not in values or not np.isfinite(values[name]) for name in features):
                continue
            inputs.append({name: float(values[name]) for name in features})
            row_indices.append(idx)
        except (ValueError, TypeError, KeyError, OverflowError):
            continue
    if not inputs:
        return result, "Gerekli hava, arazi veya zaman eşleşmeleri eksik; model skoru hesaplanmadı."
    try:
        probabilities = model.predict_proba(pd.DataFrame(inputs, columns=list(features)))[:, 1]
    except Exception as exc:
        return result, f"Model girdisiyle tahmin yapılamadı: {type(exc).__name__}: {exc}"
    if len(probabilities) < 2:
        return result, "Görünümde sıralama için en az iki uygun S-NPP/NOAA-20 noktası gerekiyor."
    percentiles = pd.Series(probabilities).rank(method="average", pct=True).to_numpy() * 100.0
    for idx, percentile in zip(row_indices, percentiles):
        if np.isfinite(percentile):
            result.loc[idx, "xgboost_rank_percentile"] = float(percentile)
    return result, "Sıralama yalnız bu görünümdeki uygun noktalar arasında hesaplandı; olasılık değildir."


def sidebar_inputs() -> tuple[str, float, float, float, bool, str, float, float, float, bool]:
    st.sidebar.markdown("## MaydonOs")
    st.sidebar.caption("Diyarbakır saha izleme")
    st.sidebar.markdown("#### Bölge seçimi")
    if not SYSTEM_CA_STORE_ENABLED:
        st.sidebar.info("Windows sistem sertifikaları için requirements.txt bağımlılıklarını kurun.")
    district_names = ["Tüm Kent", "Bismil Ovası", "Çınar", "Sur / Merkez", "Kayapınar", "Ergani"]
    district_lookup = {
        "Tüm Kent": "Tüm Kent", "Bismil Ovası": "Bismil", "Çınar": "Çınar",
        "Sur / Merkez": "Sur", "Kayapınar": "Kayapınar", "Ergani": "Ergani",
    }
    district = st.sidebar.selectbox("İlçe", district_names, index=1)
    district_key = district_lookup[district]
    lat, lon, default_radius = DISTRICTS[district_key]
    manual = st.sidebar.toggle("Koordinatı elle seç", value=False)
    if manual:
        lat = st.sidebar.number_input("Enlem", min_value=35.0, max_value=43.0, value=lat, step=0.001, format="%.4f")
        lon = st.sidebar.number_input("Boylam", min_value=25.0, max_value=46.0, value=lon, step=0.001, format="%.4f")
    radius = st.sidebar.slider("Harita alanı yarıçapı (°)", 0.02, 0.75, float(default_radius), 0.01)
    view_mode = st.sidebar.segmented_control(
        "Uydu görünümü",
        ["Gündüz", "Gece"],
        default="Gündüz",
        format_func=lambda value: "Gündüz riski" if value == "Gündüz" else "Gece geçişi",
        width="stretch",
    )
    night = view_mode == "Gece"
    st.sidebar.divider()
    firms_key = secret_or_environment("FIRMS_MAP_KEY")
    st.sidebar.caption(
        "NASA FIRMS ve otomatik arazi sınıflandırma anahtarları uygulama ayarlarından okunur."
    )
    vision_opt_in = st.sidebar.checkbox(
        "Görsel arazi sınıflandırmasını aç",
        value=False,
        help=(
            "Açıldığında en fazla 8 gerçek sıcak noktanın koordinatı uydu görüntüsü almak için Esri'ye, "
            "görüntüler ise arazi türünü sınıflandırmak için ayarladığınız görsel analiz sağlayıcısına gönderilir. "
            "Yanık izi ya da yangın teyidi yapılmaz. Bu ayarı istediğiniz an kapatabilirsiniz."
        ),
    )
    st.sidebar.caption("Bu uygulama konum, kamera veya mikrofon izni istemez. Görsel sınıflandırma isteğe bağlıdır.")
    st.sidebar.caption(
        "Hava, FIRMS ve harita verileri seçtiğiniz bölgeye göre ilgili veri sağlayıcılarından alınır. "
        "Tarayıcı konumunuza veya mikrofonunuza erişilmez."
    )
    with st.sidebar.expander("CFFDRS başlangıç kodları", expanded=False):
        st.caption("Dünkü kodlarınız varsa girin. Değerler bilinmiyorsa başlangıç varsayımlarıdır; yerel ölçüm değildir.")
        prev_ffmc = st.number_input("Önceki FFMC", min_value=0.0, max_value=101.0, value=85.0, step=1.0)
        prev_dmc = st.number_input("Önceki DMC", min_value=0.0, max_value=500.0, value=6.0, step=1.0)
        prev_dc = st.number_input("Önceki DC", min_value=0.0, max_value=1000.0, value=15.0, step=1.0)
    if st.sidebar.button("Verileri yenile", icon=":material/refresh:", width="stretch"):
        fetch_open_meteo_weather.clear()
        clear_firms_hotspots_cache()
        st.rerun()
    return (
        district, float(lat), float(lon), float(radius), night, firms_key,
        prev_ffmc, prev_dmc, prev_dc, bool(vision_opt_in),
    )


def filter_night_window(hotspots: pd.DataFrame) -> pd.DataFrame:
    """Keep nighttime detections from 01:30 to 03:30 Diyarbakır local time."""
    if hotspots.empty:
        return hotspots.copy()

    def in_window(row: pd.Series) -> bool:
        if str(row.get("daynight", "")).upper() != "N":
            return False
        try:
            acquisition_time = int(float(row["acq_time"]))
            utc_stamp = datetime.strptime(
                f"{row['acq_date']} {acquisition_time:04d}", "%Y-%m-%d %H%M"
            ).replace(tzinfo=timezone.utc)
            local_stamp = utc_stamp.astimezone(timezone(timedelta(hours=3)))
            local_minutes = local_stamp.hour * 60 + local_stamp.minute
            return 90 <= local_minutes <= 210
        except (ValueError, TypeError, KeyError, OverflowError):
            return False

    return hotspots.loc[hotspots.apply(in_window, axis=1)].reset_index(drop=True)


def filter_recent_firms_points(hotspots: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
    """Keep only last-N-hour real events for the dashboard while retaining demo labels."""
    if hotspots.empty:
        return hotspots.copy()
    frame = hotspots.copy()
    demo = frame.get("is_demo", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    if not {"acq_date", "acq_time"}.issubset(frame.columns):
        return frame.loc[demo].reset_index(drop=True)
    time_text = pd.to_numeric(frame["acq_time"], errors="coerce").map(
        lambda value: f"{int(value):04d}" if pd.notna(value) else ""
    )
    stamps = pd.to_datetime(
        frame["acq_date"].astype(str) + " " + time_text,
        format="%Y-%m-%d %H%M", errors="coerce", utc=True,
    )
    now = pd.Timestamp.now(tz="UTC")
    recent_real = stamps.between(now - pd.Timedelta(hours=hours), now, inclusive="both")
    return frame.loc[demo | recent_real].reset_index(drop=True)


def _destination_point(lat: float, lon: float, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    earth_km = 6371.0088
    bearing = np.radians(bearing_deg)
    lat1, lon1 = np.radians(lat), np.radians(lon)
    angular = distance_km / earth_km
    lat2 = np.arcsin(np.sin(lat1) * np.cos(angular) + np.cos(lat1) * np.sin(angular) * np.cos(bearing))
    lon2 = lon1 + np.arctan2(
        np.sin(bearing) * np.sin(angular) * np.cos(lat1),
        np.cos(angular) - np.sin(lat1) * np.sin(lat2),
    )
    return float(np.degrees(lat2)), float((np.degrees(lon2) + 540) % 360 - 180)


def smoke_plume_polygon(
    lat: float, lon: float, wind_direction_from: float, wind_speed_kmh: float
) -> list[tuple[float, float]] | None:
    """Create a simple wind-aligned screening cone, not a dispersion forecast."""
    values = (float(lat), float(lon), float(wind_direction_from), float(wind_speed_kmh))
    if not all(np.isfinite(values)) or not 0 <= values[2] <= 360 or values[3] < 1:
        return None
    downwind = (values[2] + 180) % 360
    length_km = float(np.clip(0.5 + values[3] * 0.12, 0.5, 8.0))
    half_width_km = length_km * 0.28
    left_bearing, right_bearing = (downwind - 90) % 360, (downwind + 90) % 360
    origin_left = _destination_point(values[0], values[1], left_bearing, 0.08)
    origin_right = _destination_point(values[0], values[1], right_bearing, 0.08)
    tip = _destination_point(values[0], values[1], downwind, length_km)
    tip_left = _destination_point(*tip, left_bearing, half_width_km)
    tip_right = _destination_point(*tip, right_bearing, half_width_km)
    return [origin_left, tip_left, tip, tip_right, origin_right]


def estimate_frp_emissions(hotspots: pd.DataFrame, assumed_duration_seconds: float = 60.0) -> dict[str, float | int]:
    """Illustrative one-minute FRP-equivalent emissions; not event totals.

    Uses biomass = 0.368 kg/MJ × FRE, plus crop-residue factors of 1.351 kg
    CO2/kg and 10.6 g PM2.5/kg. FIRMS supplies instantaneous FRP, not duration.
    """
    if not np.isfinite(assumed_duration_seconds) or assumed_duration_seconds <= 0:
        raise ValueError("Varsayılan FRP süresi pozitif ve sonlu olmalı.")
    if hotspots.empty or "frp" not in hotspots:
        return {"detections": 0, "fre_mj": 0.0, "biomass_kg": 0.0, "co2_kg": 0.0, "pm25_kg": 0.0}
    data = hotspots.copy()
    data["frp"] = pd.to_numeric(data["frp"], errors="coerce")
    data["latitude"] = pd.to_numeric(data["latitude"], errors="coerce")
    data["longitude"] = pd.to_numeric(data["longitude"], errors="coerce")
    data = data.loc[
        data["frp"].gt(0) & data["latitude"].notna() & data["longitude"].notna()
    ].copy()
    if data.empty:
        return {"detections": 0, "fre_mj": 0.0, "biomass_kg": 0.0, "co2_kg": 0.0, "pm25_kg": 0.0}
    data["_lat"] = data["latitude"].round(4)
    data["_lon"] = data["longitude"].round(4)
    data["_time"] = (
        pd.to_numeric(data["acq_time"], errors="coerce")
        if "acq_time" in data else np.nan
    )
    if "acq_date" in data:
        keys = ["_lat", "_lon", "acq_date", "_time"]
        data = data.groupby(keys, dropna=False, as_index=False)["frp"].max()
    fre_mj = float(data["frp"].sum() * assumed_duration_seconds)
    biomass_kg = fre_mj * 0.368
    return {
        "detections": int(len(data)), "fre_mj": fre_mj,
        "biomass_kg": biomass_kg, "co2_kg": biomass_kg * 1.351,
        "pm25_kg": biomass_kg * 0.0106,
    }


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = np.radians([a[0], a[1], b[0], b[1]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1))))


def patrol_route(
    hotspots: pd.DataFrame, start: tuple[float, float], max_stops: int = 8
) -> tuple[list[tuple[float, float]], float]:
    """Approximate nearest-neighbour route through unverified high-priority points."""
    if hotspots.empty:
        return [], 0.0
    candidates = hotspots.loc[~hotspots.get("is_demo", pd.Series(False, index=hotspots.index)).fillna(False).astype(bool)].copy()
    if candidates.empty:
        return [], 0.0
    candidates["_triage"] = candidates.apply(lambda row: anomaly_priority(row)[0], axis=1)
    model_scores = (
        pd.to_numeric(candidates["xgboost_rank_percentile"], errors="coerce")
        if "xgboost_rank_percentile" in candidates else pd.Series(np.nan, index=candidates.index)
    )
    candidates["_model_score"] = model_scores
    candidates = candidates.loc[candidates["_triage"].ge(50) | candidates["_model_score"].ge(60)]
    candidates = candidates.sort_values(
        ["_triage", "_model_score"], ascending=False, na_position="last"
    ).head(max_stops)
    remaining = [(float(row.latitude), float(row.longitude)) for row in candidates.itertuples()]
    if not remaining:
        return [], 0.0
    route = [tuple(map(float, start))]
    total_km = 0.0
    while remaining:
        current = route[-1]
        next_index = min(range(len(remaining)), key=lambda i: _haversine_km(current, remaining[i]))
        next_point = remaining.pop(next_index)
        total_km += _haversine_km(current, next_point)
        route.append(next_point)
    return route, total_km


def make_map(
    center: tuple[float, float],
    bbox: tuple[float, float, float, float],
    ffmc: float | None,
    hotspots: pd.DataFrame,
    night_mode: bool,
    wind_direction: float | None = None,
    wind_speed: float | None = None,
) -> folium.Map:
    fmap = folium.Map(location=center, zoom_start=10, tiles=None, control_scale=True)
    if night_mode:
        folium.TileLayer(
            tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
            attr="© OpenStreetMap contributors © CARTO",
            name="Koyu harita",
        ).add_to(fmap)
    else:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles © Esri, Maxar, Earthstar Geographics",
            name="Esri World Imagery",
        ).add_to(fmap)
    folium.raster_layers.WmsTileLayer(
        url="https://copernicus.discomap.eea.europa.eu/arcgis/services/Corine/CLC2018_WM/MapServer/WMSServer",
        layers="0", name="Copernicus CORINE arazi örtüsü (2018)", fmt="image/png",
        transparent=True,
        attr="© European Environment Agency / Copernicus Land Monitoring Service · CORINE Land Cover 2018",
        show=False, overlay=True, control=True,
    ).add_to(fmap)
    west, south, east, north = bbox
    folium.Rectangle(
        bounds=[[south, west], [north, east]], color="#d5a63e", weight=2,
        fill=False, tooltip="Seçili FIRMS arama alanı",
    ).add_to(fmap)
    if ffmc is not None:
        label, color, _ = classify_ffmc(ffmc)
        folium.Circle(
            location=center,
            radius=8000,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.13,
            tooltip=f"Seçili koordinatın FFMC göstergesi: {label} ({ffmc:.1f}) • alan içi ölçüm değildir",
        ).add_to(fmap)
        folium.CircleMarker(
            location=center, radius=7, color="#f3c84b", fill=True,
            fill_opacity=1, tooltip="Open-Meteo hava koordinatı",
        ).add_to(fmap)

    layer = folium.FeatureGroup(name="🌡️ FIRMS termal noktaları", show=True)
    for idx, row in hotspots.reset_index(drop=True).iterrows():
        is_demo = bool(row.get("is_demo", False))
        score, priority, color = anomaly_priority(row)
        anomaly_score = pd.to_numeric(row.get("anomaly_score"), errors="coerce")
        anomaly_label = (
            f" • Bölgesel değişim puanı {float(anomaly_score):.0f}/100"
            if pd.notna(anomaly_score) else ""
        )
        osm_candidate = str(row.get("osm_candidate", ""))
        classification, point_color = classify_hotspot_tr(row, ffmc)
        saved_vision = st.session_state.get("vision_classification_results", {}).get(
            hotspot_analysis_id(row), {}
        ) if not is_demo else {}
        vision_analysis = saved_vision.get("analysis") if saved_vision else None
        if vision_analysis:
            classification = LAND_COVER_LABELS.get(
                vision_analysis.get("land_cover", "uncertain"), "Görüntüden ayırt edilemiyor"
            )
            point_color = {
                "agriculture": "#c88918", "industrial": "#59636e",
                "forest": "#2e7d4f", "other": "#55758d", "uncertain": "#8a7e69",
            }.get(vision_analysis.get("land_cover", "uncertain"), "#8a7e69")
        if osm_candidate and osm_candidate != "Bağlam yetersiz / belirsiz":
            anomaly_label += f" • {osm_candidate}"
        daynight = str(row.get("daynight", "")).upper()
        if night_mode and daynight == "N" and not is_demo:
            point_color = "#b329e6"
        pulse = "box-shadow:0 0 0 8px #a634ff66,0 0 20px #ff234dcc;" if night_mode and daynight == "N" else ""
        title = "SENTETİK DEMO • gerçek tespit değil" if is_demo else "NASA FIRMS uydu bildirimi"
        anomaly_popup = (
            f"Bölgesel değişim puanı: {float(anomaly_score):.0f}/100<br>"
            if pd.notna(anomaly_score) else ""
        )
        osm_popup = f"OSM bağlamı: {escape(osm_candidate)}<br>" if osm_candidate and not is_demo else ""
        clc_code = row.get("corine_code")
        clc_popup = (
            f"CORINE 2018: {escape(str(clc_code))} · {escape(str(row.get('corine_class', '')))}<br>"
            if clc_code is not None and not pd.isna(clc_code) else "CORINE sınıfı: bulunamadı<br>"
        )
        frp_value = pd.to_numeric(row.get("frp"), errors="coerce")
        frp_popup = f"FRP: {float(frp_value):.2f} MW<br>" if pd.notna(frp_value) else "FRP: veri yok<br>"
        model_score = pd.to_numeric(row.get("xgboost_rank_percentile"), errors="coerce")
        model_popup = (
            f"Adaylar arası model sırası: {float(model_score):.1f} yüzdelik<br>"
            if pd.notna(model_score) else ""
        )
        imagery_link = ""
        if not is_demo:
            try:
                snapshot_url = escape(
                    esri_export_url(float(row["latitude"]), float(row["longitude"]), size=400),
                    quote=True,
                )
                imagery_link = (
                    f'<br><img src="{snapshot_url}" alt="Esri World Imagery kuşbakışı görüntü" '
                    'width="240" height="240" style="object-fit:cover;border-radius:8px;margin-top:6px">'
                    '<br><small>Esri görüntüsü olay anına ait olmayabilir.</small>'
                )
            except (ValueError, TypeError, KeyError):
                imagery_link = ""
        if vision_analysis:
            vision_popup = (
                "<hr><b>Görsel arazi sınıfı:</b> "
                f"{escape(LAND_COVER_LABELS.get(vision_analysis.get('land_cover', ''), 'Bilinmiyor'))}<br>"
                f"Yüzey görünümü: {escape(SURFACE_STATE_LABELS.get(vision_analysis.get('surface_state', 'uncertain'), 'Belirsiz'))}<br>"
                f"{escape(vision_analysis.get('description', ''))}<br>"
                f"Görüntü ipuçları: {escape(vision_analysis.get('visual_evidence', ''))}<br>"
                f"Nitel güven düzeyi: {escape(vision_analysis.get('confidence', 'low'))}<br>"
                "Yangın/anız yakımı teyidi değildir."
            )
        elif not is_demo:
            vision_popup = "<br><small>Görsel arazi sınıflandırması henüz tamamlanmadı.</small>"
        else:
            vision_popup = ""
        icon_emoji = (
            {"agriculture": "🌾", "industrial": "🏭", "forest": "🌲", "other": "🟠", "uncertain": "❔"}
            .get(vision_analysis.get("land_cover"), "🔥") if vision_analysis
            else ("🏭" if "Sanayi" in classification else ("🌲" if "Orman" in classification else "🔥"))
        )
        folium.Marker(
            location=(float(row["latitude"]), float(row["longitude"])),
            icon=folium.DivIcon(
                icon_size=(40, 40), icon_anchor=(20, 20),
                html=(f'<div style="width:36px;height:36px;border-radius:50%;background:{point_color};'
                      f'border:3px solid #fff;{pulse}display:flex;align-items:center;justify-content:center;'
                      f'color:#fff;font:700 16px Arial">{icon_emoji}</div>'),
            ),
            tooltip=f"MAYDONOS_FIRMS:{idx} • {title} • {classification} • İnceleme puanı {score}/100{anomaly_label}",
            popup=folium.Popup(
                f"<b>{escape(title)}</b><br>{escape(str(row.get('acq_date', '—')))} "
                f"• {escape(str(row.get('acq_time', '—')))} UTC<br>"
                f"İnceleme puanı: {score}/100 · {escape(priority)}<br>"
                f"{model_popup}"
                f"{anomaly_popup}"
                f"Sınıf: {escape(classification)}<br>"
                f"FFMC: {f'{float(ffmc):.1f}' if ffmc is not None else 'veri yok'} · {frp_popup}"
                f"{clc_popup}"
                f"{osm_popup}"
                f"{imagery_link}"
                f"{vision_popup}"
                "Bu sınıf yangın nedenini veya kaçak yakımı doğrulamaz.",
                max_width=290,
            ),
        ).add_to(layer)
    layer.add_to(fmap)

    if wind_direction is not None and wind_speed is not None and not hotspots.empty:
        plume_layer = folium.FeatureGroup(name="💨 Tahmini rüzgâr altı duman konileri", show=True)
        for _, row in hotspots.iterrows():
            if bool(row.get("is_demo", False)):
                continue
            polygon = smoke_plume_polygon(
                float(row["latitude"]), float(row["longitude"]), wind_direction, wind_speed
            )
            if polygon:
                folium.Polygon(
                    locations=polygon, color="#bd6a2d", weight=1,
                    fill=True, fill_color="#df9b56", fill_opacity=0.22,
                    tooltip=f"Rüzgâr yönüne göre basit koni · yön {wind_direction:.0f}° · hız {wind_speed:.1f} km/sa · gözlenen duman değildir",
                ).add_to(plume_layer)
        plume_layer.add_to(fmap)

    route, route_km = patrol_route(hotspots, center)
    if len(route) >= 2:
        route_layer = folium.FeatureGroup(name="🌙 Yaklaşık devriye rota önerisi", show=True)
        folium.PolyLine(
            route, color="#344fba", weight=4, opacity=0.8,
            tooltip=f"En yakın komşu yaklaşık rotası · {len(route)-1} sıcak nokta · {route_km:.1f} km düz çizgi toplamı; yol rotası değildir",
        ).add_to(route_layer)
        for stop_number, point in enumerate(route[1:], start=1):
            folium.CircleMarker(
                point, radius=6, color="#fff", weight=2, fill=True,
                fill_color="#344fba", fill_opacity=1,
                tooltip=f"Yaklaşık kontrol sırası {stop_number}",
            ).add_to(route_layer)
        route_layer.add_to(fmap)
    folium.LayerControl(collapsed=True).add_to(fmap)
    fmap.fit_bounds([[south, west], [north, east]])
    return fmap


def hotspot_analysis_id(row: pd.Series) -> str:
    # The Esri basemap is identical for repeated satellite detections at the
    # same ~11 m coordinate cell; reuse its single scene classification.
    return f"{float(row['latitude']):.4f}|{float(row['longitude']):.4f}"


def auto_classify_hotspots(hotspots: pd.DataFrame, api_key: str) -> None:
    """Classify only the top eight real detections, concurrently and once per scene."""
    if hotspots.empty or not api_key.strip():
        return
    results = st.session_state.setdefault("vision_classification_results", {})
    real = hotspots.loc[
        ~hotspots.get("is_demo", pd.Series(False, index=hotspots.index)).fillna(False).astype(bool)
    ].copy()
    if real.empty:
        return
    real["_priority"] = real.apply(lambda row: anomaly_priority(row)[0], axis=1)
    real = real.sort_values(["_priority", "frp"], ascending=False, na_position="last").head(8)

    pending: list[pd.Series] = []
    for _, row in real.iterrows():
        item_id = hotspot_analysis_id(row)
        if item_id in results:
            continue
        code = str(row.get("corine_code", "")).split(".")[0]
        if code in {"121", "131", "133"}:
            results[item_id] = {
                "skipped": True,
                "reason": f"CORINE {code}: sanayi/maden/inşaat sınıfı için görsel analiz atlandı.",
                "attempted": True,
            }
        else:
            pending.append(row)
    if not pending:
        return

    def classify_one(row: pd.Series) -> tuple[str, dict[str, object]]:
        item_id = hotspot_analysis_id(row)
        try:
            image_bytes = fetch_esri_snapshot(float(row["latitude"]), float(row["longitude"]), 400)
            analysis = analyze_overhead_scene(api_key, image_bytes)
            return item_id, {"analysis": analysis, "image_bytes": image_bytes, "attempted": True}
        except Exception as exc:
            message = str(exc).replace(api_key, "[gizlendi]")[:400]
            return item_id, {"error": message, "attempted": True}

    with st.status(
        f"En yüksek öncelikli {len(pending)} uydu noktası arazi bağlamı için inceleniyor…",
        expanded=True,
    ) as status:
        progress = st.progress(0.0, text="Görüntüler paralel alınıyor ve sınıflandırılıyor…")
        completed = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(classify_one, row) for row in pending]
            for future in as_completed(futures):
                item_id, record = future.result()
                results[item_id] = record
                completed += 1
                progress.progress(completed / len(pending), text=f"Tamamlanan: {completed}/{len(pending)}")
        failed = sum(bool(results.get(hotspot_analysis_id(row), {}).get("error")) for row in pending)
        progress.empty()
        status.update(
            label=f"Görsel arazi sınıflandırması tamamlandı · {len(pending) - failed} başarılı, {failed} başarısız",
            state="complete" if failed == 0 else "error",
            expanded=False,
        )

def render_hotspot_card(row: pd.Series, number: int, vision_api_key: str = "") -> None:
    lat, lon = float(row["latitude"]), float(row["longitude"])
    is_demo = bool(row.get("is_demo", False))
    address = reverse_geocode(round(lat, 6), round(lon, 6)) if not is_demo else None
    confidence = str(row.get("confidence", "bilinmiyor")).strip().lower()
    confidence_label = {"h": "Yüksek", "n": "Normal", "l": "Düşük", "high": "Yüksek", "nominal": "Normal", "low": "Düşük"}.get(confidence, confidence.title())
    daynight = "Gece" if str(row.get("daynight", "")).upper() == "N" else "Gündüz"
    frp = pd.to_numeric(row.get("frp"), errors="coerce")
    brightness = pd.to_numeric(row.get("bright_ti4"), errors="coerce")
    score, priority, score_color = anomaly_priority(row)
    xgb_score = pd.to_numeric(row.get("xgboost_rank_percentile"), errors="coerce")
    with st.container(border=True):
        st.subheader(f"{'Demo noktası' if is_demo else 'Uydu sıcak nokta bildirimi'} · #{number}")
        if is_demo:
            st.error("SENTETİK DEMO VERİSİ — gerçek NASA FIRMS tespiti değildir.", icon="⚠️")
        else:
            st.warning("Termal anomali yangının doğrulandığı anlamına gelmez; sahada kontrol gerekir.", icon="⚠️")
        st.markdown(f"**{priority} · {score}/100**", help="İnceleme önceliği puanıdır; yangın olasılığı değildir.")
        if pd.notna(xgb_score):
            st.metric(
                "Görünüm içi aday sırası",
                f"{float(xgb_score):.1f}. yüzdelik",
                help="Yalnız aynı harita görünümünde skoru hesaplanan sıcak nokta adayları arasındaki göreli sıralamadır; olasılık değildir.",
            )
        classification, _ = classify_hotspot_tr(row, st.session_state.get("current_ffmc"))
        vision_record = st.session_state.get("vision_classification_results", {}).get(
            hotspot_analysis_id(row), {}
        ) if not is_demo else {}
        vision_result = vision_record.get("analysis") if vision_record else None
        if vision_result:
            st.markdown(
                f"**Görsel arazi sınıfı:** {LAND_COVER_LABELS.get(vision_result['land_cover'], vision_result['land_cover'])}"
            )
            st.caption(
                "Görüntüdeki yüzey görünümü: "
                + SURFACE_STATE_LABELS.get(vision_result.get("surface_state", "uncertain"), "Belirsiz")
            )
            st.write(vision_result["description"])
            st.caption(
                f"Görüntü ipuçları: {vision_result['visual_evidence']} · "
                f"Modelin nitel güven düzeyi: {vision_result['confidence']}"
            )
            st.caption(vision_result["limitations"])
            st.caption("Arazi bağlamı sınıflandırmasıdır; yangını veya anız yakımını doğrulamaz.")
        else:
            st.markdown(f"**Harita bağlamı sınıf adayı:** {escape(classification)}")
        if not is_demo and pd.notna(row.get("osm_candidate")):
            st.markdown(f"**Yakın çevre ipucu:** {escape(str(row.get('osm_candidate')))}")
            context_text = str(row.get("osm_nearby", "")).strip()
            if context_text:
                st.caption(context_text)
        clc_code = row.get("corine_code")
        if clc_code is not None and not pd.isna(clc_code):
            st.caption(f"CORINE 2018 bağlamı: {clc_code} · {row.get('corine_class', 'Sınıf adı yok')}")
        else:
            st.caption("CORINE sınıfı bu nokta için alınamadı; arazi sınıfı belirsiz.")
        st.markdown("**Yakın konum**")
        st.write(address or "Adres bilgisi alınamadı; aşağıdaki koordinat ve harita bağlantısı kullanılabilir.")
        c1, c2 = st.columns(2)
        c1.metric("Koordinat", f"{lat:.5f}, {lon:.5f}")
        raw_time = pd.to_numeric(row.get("acq_time"), errors="coerce")
        if pd.notna(raw_time):
            stamp = f"{int(raw_time):04d}"
            time_text = f"{row.get('acq_date', '—')} · {stamp[:2]}:{stamp[2:]} UTC"
        else:
            time_text = f"{row.get('acq_date', '—')} · — UTC"
        c2.metric("Uydu ölçüm zamanı", time_text)
        c3, c4, c5 = st.columns(3)
        c3.metric("Uydu güven düzeyi", "Demo" if is_demo else confidence_label)
        c4.metric("Gündüz / gece", daynight)
        c5.metric("FRP", f"{frp:.2f} MW" if pd.notna(frp) else "Veri yok")
        if pd.notna(brightness):
            st.caption(f"VIIRS Ti4 parlaklık ölçümü: {brightness:.1f} K")
        st.link_button("Konumu haritada aç", f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map=16/{lat:.6f}/{lon:.6f}", width="content")
        if not is_demo:
            st.markdown("**Kuşbakışı uydu görüntüsü ve otomatik arazi sınıflandırması**")
            st.caption(
                "Esri görüntüsünün çekim tarihi FIRMS ölçüm tarihiyle aynı olmayabilir; bu görüntü olayın kendisine dair yorum üretmekte kullanılmaz."
            )
            image_slot = st.empty()
            result_key = hotspot_analysis_id(row)
            saved = st.session_state.get("vision_classification_results", {}).get(result_key)
            image_bytes = saved.get("image_bytes") if saved else None
            if image_bytes is None and not (saved and saved.get("error")):
                try:
                    with st.spinner("Esri World Imagery kuşbakışı görüntüsü alınıyor…"):
                        image_bytes = fetch_esri_snapshot(lat, lon, 400)
                except (requests.RequestException, ValueError, OSError) as exc:
                    st.warning(f"Uydu görüntüsü alınamadı: {exc}")
            if image_bytes:
                image_slot.image(image_bytes, caption="Esri World Imagery · 400 × 400 piksel · olay zamanı garantisi yok", width=400, alt="Seçili FIRMS koordinatının Esri kuşbakışı görüntüsü")
            if saved and saved.get("error"):
                st.warning(f"Görsel sınıflandırma yapılamadı: {saved['error']}")
            elif saved and saved.get("skipped"):
                st.info(str(saved.get("reason", "Bu arazi sınıfı için görsel analiz atlandı.")))
            elif not vision_api_key:
                st.info("Otomatik görsel sınıflandırma kapalı; VISION_API_KEY uygulama ayarlarında tanımlı değil. Kural tabanlı OSM/CORINE sınıf adayı gösteriliyor.")


def render_live_map(
    lat: float,
    lon: float,
    bbox: tuple[float, float, float, float],
    ffmc: float | None,
    hotspots: pd.DataFrame,
    night: bool,
    wind_direction: float | None = None,
    wind_speed: float | None = None,
    vision_api_key: str = "",
) -> None:
    map_state = st_folium(
        make_map((lat, lon), bbox, ffmc, hotspots, night, wind_direction, wind_speed),
        height=600,
        width="stretch",
        returned_objects=["last_object_clicked_tooltip"],
        key="maydonos_firms_map",
    )
    clicked = (map_state or {}).get("last_object_clicked_tooltip")
    if isinstance(clicked, str) and clicked.startswith("MAYDONOS_FIRMS:"):
        try:
            st.session_state["selected_firms_hotspot"] = int(clicked.split(":", 1)[1].split(maxsplit=1)[0])
        except ValueError:
            pass
    selected_index = st.session_state.get("selected_firms_hotspot")
    if hotspots.empty:
        st.session_state.pop("selected_firms_hotspot", None)
    elif isinstance(selected_index, int) and 0 <= selected_index < len(hotspots):
        render_hotspot_card(
            hotspots.reset_index(drop=True).iloc[selected_index],
            selected_index + 1,
            vision_api_key,
        )


def render_sensitivity_simulator(
    current: pd.Series | None,
    rain24: float | None,
    live_indices: dict[str, float] | None,
    lat: float,
    prev_ffmc: float,
    prev_dmc: float,
    prev_dc: float,
) -> None:
    st.subheader("Hava koşulu denemesi")
    st.caption("Sıcaklık, nem veya rüzgârı değiştirince CFFDRS FFMC → ISI → FWI zinciri yeniden hesaplanır.")
    if current is None or rain24 is None or live_indices is None:
        st.warning("Canlı hava ve eksiksiz önceki 24 saat yağış verisi olmadan simülasyon hesaplanmaz.")
        return

    live_temp = float(current["temperature_2m"])
    live_rh = float(current["relative_humidity_2m"])
    live_wind = float(current["wind_speed_10m"])
    s1, s2, s3 = st.columns(3)
    temperature = s1.slider(
        "Sıcaklık (°C)", 15, 50, min(50, max(15, int(round(live_temp)))), 1,
        help=f"Canlı Open-Meteo değeri: {live_temp:.1f} °C",
    )
    humidity = s2.slider(
        "Nispi nem (%)", 10, 90, min(90, max(10, int(round(live_rh)))), 1,
        help=f"Canlı Open-Meteo değeri: %{live_rh:.0f}",
    )
    wind = s3.slider(
        "Rüzgâr hızı (km/sa)", 0, 60, min(60, max(0, int(round(live_wind)))), 1,
        help=f"Canlı Open-Meteo değeri: {live_wind:.1f} km/sa",
    )
    try:
        simulated_indices = calculate_cffdrs_system(
            temp=temperature,
            rh=humidity,
            ws=wind,
            prec=rain24,
            lat=lat,
            month=pd.Timestamp(current["time"]).month,
            prev_ffmc=prev_ffmc,
            prev_dmc=prev_dmc,
            prev_dc=prev_dc,
        )
    except (RuntimeError, ValueError) as exc:
        st.error(f"Simülasyon hesaplanamadı: {exc}")
        return

    label, _, emoji = classify_ffmc(simulated_indices["ffmc"])
    m1, m2 = st.columns(2)
    m1.metric("Canlı FWI", f"{live_indices['fwi']:.1f}")
    m2.metric("Simüle FWI", f"{simulated_indices['fwi']:.1f}", f"{simulated_indices['fwi'] - live_indices['fwi']:+.1f}", delta_color="inverse")
    st.info(f"{emoji} Simüle FFMC: **{simulated_indices['ffmc']:.1f}** ({label}). Yağış: önceki 24 saat **{rain24:.1f} mm**.")
    comparison = pd.DataFrame(
        {"Senaryo": ["Canlı hava", "Simüle hava"], "FWI": [live_indices["fwi"], simulated_indices["fwi"]]}
    )
    st.bar_chart(comparison, x="Senaryo", y="FWI", y_label="FWI (indeks)", alt="Canlı ve simüle edilmiş CFFDRS FWI indekslerinin karşılaştırması")
    component_table = pd.DataFrame([
        {"Bileşen": key.upper(), "Canlı": live_indices[key], "Simüle": simulated_indices[key], "Fark": simulated_indices[key] - live_indices[key]}
        for key in ("ffmc", "dmc", "dc", "isi", "bui", "fwi")
    ])
    st.dataframe(component_table, hide_index=True, width="stretch", alt="Canlı ve simüle CFFDRS bileşen karşılaştırması")
    st.caption("FFMC sınıfları yerel yangın olasılığı değildir. FWI karşılaştırması aynı önceki kodlar ve yağışla yapılır.")


@st.fragment(run_every="15m")
def dashboard_content(
    district: str,
    lat: float,
    lon: float,
    radius: float,
    night: bool,
    firms_key: str,
    prev_ffmc: float,
    prev_dmc: float,
    prev_dc: float,
    vision_opt_in: bool,
) -> None:
    try:
        validate_coordinates(lat, lon)
        bbox = bbox_from_center(lat, lon, radius)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    weather: pd.DataFrame | None = None
    current: pd.Series | None = None
    ffmc: float | None = None
    indices: dict[str, float] | None = None
    ffmc_history: pd.DataFrame | None = None
    hourly_error: str | None = None
    rain24: float | None = None
    weather_source = "Hava verisi kullanılamıyor"
    weather_error: str | None = None
    with st.spinner("Güncel ve saatlik hava verisi alınıyor…"):
        try:
            weather, current, weather_source = fetch_open_meteo_weather(round(lat, 4), round(lon, 4))
            rain24 = previous_24h_rain(weather, pd.Timestamp(current["time"]))
            indices = calculate_cffdrs_system(
                temp=float(current["temperature_2m"]),
                rh=float(current["relative_humidity_2m"]),
                ws=float(current["wind_speed_10m"]),
                prec=rain24,
                lat=lat,
                month=pd.Timestamp(current["time"]).month,
                prev_ffmc=prev_ffmc,
                prev_dmc=prev_dmc,
                prev_dc=prev_dc,
            )
            try:
                ffmc_history = calculate_hourly_ffmc_series(weather, pd.Timestamp(current["time"]), initial_ffmc=prev_ffmc)
                ffmc = float(ffmc_history["ffmc_hourly"].iloc[-1])
            except (RuntimeError, ValueError, TypeError, KeyError) as exc:
                # Daily FFMC remains available; the UI labels its time basis.
                ffmc = indices["ffmc"]
                hourly_error = str(exc)
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            weather_error = str(exc)

    with st.spinner("NASA FIRMS sıcak nokta tespitleri kontrol ediliyor…"):
        hotspots, firms_status = fetch_nasa_firms_hotspots(
            tuple(round(v, 4) for v in bbox), firms_key, allow_demo=True
        )
    persistence_events = hotspots.copy()
    hotspots = filter_recent_firms_points(hotspots, hours=24)
    anomaly_cells = pd.DataFrame()
    history_days = 0
    osm_status = ""
    corine_status = ""
    model_score_status = "Model skoru için gerçek FIRMS, hava ve CORINE girdileri gerekli."
    if firms_status.startswith("NASA FIRMS •"):
        complete_sensor_snapshot = "sensör kaynağı geçici olarak alınamadı" not in firms_status
        try:
            if complete_sensor_snapshot:
                history_path = PROJECT_ROOT / "data" / "firms_cell_history.csv"
                snapshot_date = datetime.now(timezone(timedelta(hours=3))).date()
                history = record_snapshot(tuple(round(v, 4) for v in bbox), hotspots, history_path, snapshot_date)
                hotspots, anomaly_cells, history_days = score_snapshot(
                    tuple(round(v, 4) for v in bbox), hotspots, history, snapshot_date
                )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            st.warning(f"Sıcak nokta geçmişi kaydedilemedi; anomali analizi gösterilmiyor. ({exc})")
        with st.spinner("Sıcak noktaların yakın çevresindeki OSM kayıtları kontrol ediliyor…"):
            hotspots, osm_status = enrich_hotspots_with_osm(hotspots)
        with st.spinner("Copernicus CORINE arazi örtüsü sorgulanıyor…"):
            hotspots, corine_status = enrich_hotspots_with_corine(hotspots)
        persistence_coverage = all(
            sensor_name in firms_status for sensor_name in ("VIIRS S-NPP", "VIIRS NOAA-20")
        )
        hotspots, model_score_status = score_hotspots_with_xgboost(
            hotspots, weather, allow_persistence=persistence_coverage,
            persistence_events=persistence_events,
        )
    wind_direction = None
    wind_speed = None
    if weather is not None and not weather.empty and current is not None:
        wind_speed_value = pd.to_numeric(current.get("wind_speed_10m"), errors="coerce")
        if pd.notna(wind_speed_value) and np.isfinite(float(wind_speed_value)):
            wind_speed = float(wind_speed_value)
        if "wind_direction_10m" in weather.columns:
            weather_times = pd.to_datetime(weather["time"], errors="coerce")
            current_time = pd.Timestamp(current["time"])
            eligible = weather.loc[weather_times.le(current_time)].copy()
            if not eligible.empty:
                direction = pd.to_numeric(eligible.iloc[-1].get("wind_direction_10m"), errors="coerce")
                if pd.notna(direction) and np.isfinite(float(direction)):
                    wind_direction = float(direction) % 360
    if night:
        hotspots = filter_night_window(hotspots)

    vision_api_key = secret_or_environment("VISION_API_KEY")
    if vision_opt_in and firms_status.startswith("NASA FIRMS •") and vision_api_key and not hotspots.empty:
        auto_classify_hotspots(hotspots, vision_api_key)

    if weather_error:
        st.error(weather_error, title="Hava verisine ulaşılamadı")
        st.info("Eksik hava verisiyle endeks hesaplanmadı. Bağlantıyı kontrol edip verileri yenileyin.")
    else:
        st.caption("Hava verisi alındı · Yeni sorgu için sol menüden verileri yenileyin.")

    st.session_state["current_ffmc"] = ffmc
    model_scores = pd.to_numeric(hotspots.get("xgboost_rank_percentile", pd.Series(dtype=float)), errors="coerce")
    valid_model_scores = model_scores.dropna()
    metric_columns = st.columns(6 if not valid_model_scores.empty else 5)
    k1, k2, k3, k4, k5 = metric_columns[:5]
    k1.metric("Hava sıcaklığı", f"{float(current['temperature_2m']):.1f} °C" if current is not None else "—")
    k2.metric("Nispi nem", f"{float(current['relative_humidity_2m']):.0f} %" if current is not None else "—")
    if ffmc is not None:
        k3.metric("Yakıt kuruluğu · FFMC", f"{ffmc:.1f}", help="Saatlik seri varsa saatlik, yoksa günlük CFFDRS adımı.")
    else:
        k3.metric("Yakıt kuruluğu · FFMC", "—")
    labels = hotspots.apply(lambda row: classify_hotspot_tr(row, ffmc)[0], axis=1) if not hotspots.empty else pd.Series(dtype=str)
    ag_candidates = int(labels.str.startswith("Tarım alanında yüksek öncelikli").sum()) if not labels.empty else 0
    industrial_contexts = int(labels.str.startswith("Sanayi / tesis çevresi").sum()) if not labels.empty else 0
    stored_vision = st.session_state.get("vision_classification_results", {})
    visible_real_ids = {
        hotspot_analysis_id(row)
        for _, row in hotspots.iterrows()
        if not bool(row.get("is_demo", False))
    }
    vision_classified = sum(
        1 for item_id in visible_real_ids
        if stored_vision.get(item_id, {}).get("analysis")
    )
    k4.metric("Tarım alanında sıcak nokta adayı", str(ag_candidates), help="FFMC ≥85 ve FRP ≥8 MW ile önceliklendirilir; saha teyidi değildir.")
    k5.metric("Görsel sınıflandırılan nokta", f"{vision_classified}", help="Esri görüntüsünde arazi sınıfı otomatik çıkarılan gerçek FIRMS nokta sayısı; yangın doğrulaması değildir.")
    if not valid_model_scores.empty:
        metric_columns[5].metric(
            "Adaylar arası göreli sıra",
            f"{valid_model_scores.mean():.1f}. yüzdelik",
            help="Yalnız bu görünümde skoru hesaplanan S-NPP/NOAA-20 noktaları arasındaki sıra; model olasılığı değildir.",
        )
    st.caption(model_score_status)
    if industrial_contexts:
        st.caption(f"OSM/CORINE bağlamında sanayi/tesis çevresinde görünen kayıt: {industrial_contexts} · bunlar doğrulanmış yanlış alarm sayılmaz.")

    if not weather_error and current is not None and ffmc is not None:
        risk_label, risk_color, emoji = classify_ffmc(ffmc)
        st.markdown(
            f"<div class='risk-summary' style='--risk-accent:{risk_color}'>"
            f"<span class='risk-dot' aria-hidden='true'></span>"
            f"<div><strong>{escape(district)} · {escape(risk_label)}</strong>"
            f"<div class='risk-detail'>Yakıt kuruluğu göstergesi (FFMC): {ffmc:.1f} · Bu değer yangın teyidi değildir.</div></div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"{weather_source} · {pd.Timestamp(current['time']):%d.%m.%Y %H:%M} yerel saat · "
            f"Önceki 24 saatin yağışı: {rain24:.1f} mm · başlangıç kodları FFMC/DMC/DC: {prev_ffmc:g}/{prev_dmc:g}/{prev_dc:g}"
        )
        if ffmc >= 85:
            st.error("Yüksek tutuşabilirlik göstergesi. Anız yakmayın; resmî yerel uyarıları izleyin.", title="Dikkat")
        elif ffmc >= 70:
            st.warning("Hava koşulları tutuşabilirliği artırıyor. Resmî yerel uyarıları takip edin.", title="Orta düzey gösterge")
        else:
            st.success("Tutuşabilirlik göstergesi düşük aralıkta.", title="Düşük gösterge")
        st.caption("FFMC eşikleri bu prototipte gösterim bandıdır; yerel yangın olasılığı veya resmî uyarı seviyesi değildir.")

    if firms_status.startswith("SENTETİK DEMO"):
        st.error(firms_status, icon="⚠️")
    elif firms_status.startswith("NASA FIRMS •"):
        st.caption(f"Uydu verisi: {firms_status}")
    elif "MAP_KEY girilmedi" in firms_status:
        st.info(firms_status)
    elif "MAP_KEY" in firms_status or "TLS" in firms_status or "zaman aşımı" in firms_status:
        st.warning(firms_status)
    else:
        st.info(firms_status)
    if corine_status:
        st.caption(corine_status)
        st.caption("CORINE 2018 geniş alan arazi örtüsü bağlamıdır (asgari haritalama birimi 25 ha); tarla sınırı veya güncel parsel doğrulaması değildir.")

    with st.expander("Bölgesel sıcak nokta değişimi", expanded=False):
        st.subheader("Bölgesel sıcak nokta değişimi")
        if not firms_status.startswith("NASA FIRMS •"):
            st.info("Anomali analizi için gerçek NASA FIRMS verisi gerekli. Demo noktaları analize katılmaz.")
        elif "sensör kaynağı geçici olarak alınamadı" in firms_status:
            st.info("FIRMS sensör kaynaklarının bir kısmına ulaşılamadı. Eksik sensör verisi anomali geçmişine yazılmadı.")
        elif history_days < 7:
            st.info(
                f"Bu arama alanında karşılaştırılabilir geçmiş {history_days}/7 gün. "
                "Uygulama gerçek FIRMS sorgularını günde bir kayıt olarak biriktiriyor; yeterli geçmiş oluşmadan anomali puanı üretmiyor."
            )
        elif not anomaly_cells.empty:
            anomaly_points = hotspots.loc[
                hotspots.get("anomaly_flag", pd.Series(False, index=hotspots.index)).fillna(False)
            ].copy()
            st.warning(
                f"Geçmiş {history_days} günlük kayıtla karşılaştırıldığında {len(anomaly_cells)} hücrede "
                "olağandışı sıcak nokta artışı görüldü. Bu, yangının doğrulandığı anlamına gelmez."
            )
            if not anomaly_points.empty:
                cols = [c for c in ("acq_date", "acq_time", "latitude", "longitude", "confidence", "frp", "anomaly_score") if c in anomaly_points]
                st.dataframe(
                    anomaly_points[cols].rename(columns={
                        "acq_date": "Tespit tarihi", "acq_time": "Saat (UTC)",
                        "latitude": "Enlem", "longitude": "Boylam", "confidence": "Uydu güveni",
                        "frp": "FRP (MW)", "anomaly_score": "Bölgesel değişim puanı (0–100)",
                    }),
                    hide_index=True, width="stretch", alt="Geçmişe göre sıra dışı sıcak nokta bildirimleri",
                )
        else:
            st.success(
                f"Karşılaştırma için {history_days} günlük geçmiş mevcut. "
                "Bugünkü veride bu alan için belirlenen anomali eşiğini aşan artış görülmedi."
            )
        st.caption(
            "Yöntem: yaklaşık 5 km hücrelerde günlük FIRMS piksel sayısı, önceki 90 günlük aynı alan geçmişinin "
            "medyan/MAD değeri ve komşu hücre artışıyla karşılaştırılır. En az 7 geçmiş gün gerekir. "
            "Puan olasılık veya doğrulanmış yangın değildir; kayıtlar data/firms_cell_history.csv dosyasında tutulur."
        )

    with st.expander("Uydu noktası ve çevre bilgileri", expanded=False):
        st.subheader("Uydu noktasının yakın çevresi")
        if osm_status:
            st.caption(osm_status)
        if not firms_status.startswith("NASA FIRMS •"):
            st.info("Yakın çevre ipuçları için gerçek FIRMS konumları gerekir.")
        elif not hotspots.empty:
            real_context = hotspots.loc[~hotspots["is_demo"].fillna(False).astype(bool)].copy()
            if not real_context.empty:
                real_context["candidate_class"] = real_context.apply(
                    lambda row: classify_hotspot_tr(row, ffmc)[0], axis=1
                )
                context_columns = [
                    column for column in (
                        "acq_date", "acq_time", "latitude", "longitude", "firms_source",
                        "frp", "xgboost_rank_percentile", "candidate_class", "osm_candidate", "osm_nearby", "corine_code", "corine_class",
                    ) if column in real_context.columns and (
                        column != "xgboost_rank_percentile"
                        or pd.to_numeric(real_context[column], errors="coerce").notna().any()
                    )
                ]
                st.dataframe(
                    real_context[context_columns].rename(columns={
                        "acq_date": "Tespit tarihi", "acq_time": "Saat (UTC)",
                        "latitude": "Enlem", "longitude": "Boylam", "firms_source": "Uydu kaynağı",
                        "frp": "FRP (MW)", "xgboost_rank_percentile": "Görünüm içi göreli sıra (yüzdelik)",
                        "candidate_class": "Açıklanabilir sınıf adayı",
                        "osm_candidate": "Harita bağlamı (aday)", "osm_nearby": "Yakındaki kayıtlar",
                        "corine_code": "CORINE kodu", "corine_class": "CORINE 2018 arazi sınıfı",
                    }),
                    hide_index=True, width="stretch", alt="NASA termal noktalarının OpenStreetMap yakın çevresi",
                )
                st.caption(
                    "Bu ipuçları OpenStreetMap’te etiketlenmiş arazi ve tesis kayıtlarına yaklaşıklıktır; "
                    "yanma nedenini doğrulamaz. Harita verisi eksik veya güncel olmayabilir."
                )
            else:
                st.info("Gerçek termal nokta bulunmadı; demo kayıtlarına OSM sınıfı atanmadı.")
        else:
            st.info("Seçili sorguda sıcak nokta yok; gösterilecek yakın çevre kaydı bulunmuyor.")

    map_tab, sensitivity_tab = st.tabs([
        "Canlı risk ve uydu haritası",
        "Hava koşulu denemesi",
    ])
    with map_tab:
        st.subheader("Bölge haritası")
        st.caption("Turuncu noktalar sentetik demo kayıtlarıdır. Duman konisi yalnız rüzgâra göre çizilmiş basit yön göstergesidir; atmosferik yayılım modeli değildir.")
        render_live_map(
            lat, lon, bbox, ffmc, hotspots, night,
            wind_direction, wind_speed, vision_api_key,
        )
    with sensitivity_tab:
        render_sensitivity_simulator(current, rain24, indices, lat, prev_ffmc, prev_dmc, prev_dc)

    st.subheader("FRP’den türetilen emisyon göstergesi")
    real_hotspots = hotspots.loc[
        ~hotspots.get("is_demo", pd.Series(False, index=hotspots.index)).fillna(False).astype(bool)
    ].copy() if not hotspots.empty else hotspots
    if real_hotspots.empty:
        st.info("Gerçek FRP ölçümü olmadığı için emisyon hesabı gösterilmiyor.")
    else:
        emissions = estimate_frp_emissions(real_hotspots, assumed_duration_seconds=60.0)
        em1, em2, em3, em4 = st.columns(4)
        em1.metric("Kullanılan benzersiz nokta", f"{emissions['detections']}")
        em2.metric("Varsayılan FRE", f"{emissions['fre_mj']:.1f} MJ")
        em3.metric("Tahmini CO₂ eşdeğeri", f"{emissions['co2_kg']:.2f} kg")
        em4.metric("Tahmini PM₂.₅", f"{emissions['pm25_kg']:.3f} kg")
        st.caption(
            "Bu yalnızca her FIRMS FRP değerinin 60 saniye sürdüğü varsayımıyla yapılan gösterim hesabıdır; "
            "uydu FRP anlık ölçümdür ve olay süresi bilinmediğinden toplam olay emisyonu değildir. "
            "Biyokütle dönüşümü Wooster vd. (2005), CO₂ ve PM₂.₅ katsayıları Li vd. (2015) ortalama tarımsal artık yakım faktörleridir; "
            "Diyarbakır’a özgü ölçüm değildir. Termal bildirim tek başına yangın teyidi sayılmaz."
        )
        with st.expander("Hesap yöntemini ve kaynakları gör"):
            st.latex(r"FRE_{60s}(MJ)=FRP(MW)\times60(s)")
            st.latex(r"M_{biyokütle}(kg)=0.368\times FRE(MJ)")
            st.latex(r"CO_2(kg)=M_{biyokütle}\times1.351;\quad PM_{2.5}(kg)=M_{biyokütle}\times0.0106")
            st.markdown(
                "- [Wooster et al. (2005), FRP/FRE ve biyokütle ilişkisi](https://doi.org/10.1029/2005JD006318)\n"
                "- [Li et al. (2015), tarımsal artık yakımında CO₂ ve PM₂.₅ emisyon faktörleri](https://doi.org/10.1016/j.atmosenv.2015.05.007)"
            )

    if indices is not None:
        with st.expander("CFFDRS bileşenlerini gör"):
            component_columns = st.columns(6)
            labels = [("FFMC", "İnce yakıt nem kodu"), ("DMC", "Organik tabaka nem kodu"),
                      ("DC", "Kuraklık kodu"), ("ISI", "İlk yayılma indeksi"),
                      ("BUI", "Yakıt birikim indeksi"), ("FWI", "Yangın-hava indeksi")]
            for column, (key, label_text) in zip(component_columns, labels):
                column.metric(key, f"{indices[key]:.1f}", help=label_text)
            st.caption("FWI bileşenleri önceki FFMC/DMC/DC kodları, güncel hava anlık değerleri ve son 24 saatin yağış toplamıyla bir adım hesaplanır.")
            st.caption("DMC/DC mevsim düzeltmesi, kütüphanenin 30°K üzerindeki yerler için tanımlı 46°K standart tablosunu kullanır; Diyarbakır için yerel uyarlama değildir.")
            if ffmc_history is not None:
                st.caption("Saatlik FFMC, Open-Meteo saatlik hava serisi üzerinde CFFDRS hourly FFMC fonksiyonuyla ardışık hesaplanır.")
            elif hourly_error:
                st.info(f"Saatlik FFMC serisi kullanılamadı; kartta günlük FFMC adımı gösteriliyor: {hourly_error}")

    left, right = st.columns(2)
    with left:
        st.subheader("24 saatlik sıcaklık ve nem trendi")
        if weather is not None:
            chart_df = weather.loc[
                weather["time"].between(
                    pd.Timestamp(current["time"]), pd.Timestamp(current["time"]) + pd.Timedelta(hours=24)
                )
            ].head(24).set_index("time")
            if chart_df.empty:
                st.info("Open-Meteo 24 saatlik ileri tahmin sağlamadı; grafik çizilmedi.")
            else:
                st.line_chart(
                    chart_df[["temperature_2m"]].rename(columns={"temperature_2m": "Sıcaklık (°C)"}),
                    y_label="°C", height=190, alt="Önümüzdeki 24 saatin saatlik sıcaklık değişimi",
                )
                st.line_chart(
                    chart_df[["relative_humidity_2m"]].rename(columns={"relative_humidity_2m": "Bağıl nem (%)"}),
                    y_label="%", height=190, alt="Önümüzdeki 24 saatin saatlik nem değişimi",
                )
        else:
            st.info("Hava verisi gelmediği için sıcaklık ve nem grafikleri çizilmedi.")
    with right:
        st.subheader("Ekipler için önerilen saha koridoru")
        st.caption("Yüksek inceleme öncelikli, henüz sahada doğrulanmamış noktalar en yakın komşu yöntemiyle sıralanır.")
        demo_rows = hotspots.loc[hotspots["is_demo"]] if not hotspots.empty else hotspots
        real_rows = hotspots.loc[~hotspots["is_demo"]] if not hotspots.empty else hotspots
        if not real_rows.empty:
            alerts = real_rows.reset_index(drop=True).copy()
            scored = alerts.apply(anomaly_priority, axis=1)
            alerts["İnceleme puanı"] = [x[0] for x in scored]
            alerts["Öncelik"] = [x[1] for x in scored]
            top_idx = int(alerts["İnceleme puanı"].idxmax())
            top = alerts.iloc[top_idx]
            st.warning(
                f"{district}: en yüksek öncelikli gerçek FIRMS termal bildirimi "
                f"{float(top['latitude']):.5f}, {float(top['longitude']):.5f} koordinatında "
                f"({int(top['İnceleme puanı'])}/100). Yetkili ekipler sahada doğrulayabilir."
            )
            route_points, route_distance = patrol_route(real_rows, (lat, lon))
            if len(route_points) >= 2:
                st.info(
                    f"Yaklaşık öneri: {len(route_points) - 1} nokta · "
                    f"başlangıç noktasından toplam {route_distance:.1f} km düz çizgi mesafesi."
                )
                st.caption("Bu en kısa yol değildir ve yol ağına göre hesaplanmaz; haritadaki çizgi kuş uçuşu yaklaşık sıradır.")
                st.dataframe(
                    pd.DataFrame([
                        {"Sıra": order, "Enlem": point[0], "Boylam": point[1]}
                        for order, point in enumerate(route_points[1:], start=1)
                    ]),
                    hide_index=True, width="stretch", alt="Yaklaşık devriye kontrol noktaları",
                )
            else:
                st.caption("Rota için 50/100 inceleme puanını aşan gerçek uydu noktası bulunmuyor.")
            st.metric("Gerçek FIRMS noktası", f"{len(real_rows)} piksel")
            with st.expander("Sıcak nokta bildirimlerini gör"):
                columns = [c for c in ("acq_date", "acq_time", "latitude", "longitude", "confidence", "frp", "xgboost_rank_percentile", "daynight", "İnceleme puanı", "Öncelik") if c in alerts and (c != "xgboost_rank_percentile" or alerts[c].notna().any())]
                st.dataframe(alerts[columns], hide_index=True, width="stretch", height=260, alt="Gerçek NASA FIRMS sıcak nokta kayıtları")
        elif not demo_rows.empty:
            st.error("FIRMS verisi alınamadı. Haritadaki sentetik demo noktaları gerçek ihbar veya devriye yönlendirmesi değildir.")
            st.metric("Sentetik demo noktası", f"{len(demo_rows)}", help="Gerçek NASA FIRMS gözlemi değildir.")
            if ffmc is not None and ffmc >= 85:
                st.warning(
                    f"{district} bölgesinde FFMC {ffmc:.1f} ile yüksek meteorolojik gösterge var. "
                    "Bu, uydu tespiti veya kaçak yakım kanıtı değildir; ekipler kendi operasyonel prosedürlerine göre değerlendirsin."
                )
                st.caption("Gece görünümü penceresi 01:30–03:30 yerel saattir. Bu ekran gerçek yol rotası hesaplamaz.")
        else:
            st.info("Seçili alanda gerçek uydu bildirimi bulunmadı; otomatik denetim koridoru üretilmedi.")
            st.metric("Gerçek FIRMS noktası", "0")
            if ffmc is not None and ffmc >= 85:
                st.warning(
                    f"{district} için FFMC {ffmc:.1f}: meteorolojik tutuşabilirlik göstergesi yüksek. "
                    "Bu bir yangın tespiti değildir; saha kontrolü kararı yetkili ekiplerindir."
                )
                st.caption("Gece görünümü penceresi 01:30–03:30 yerel saattir; gerçek rota hesaplanmamıştır.")

    with st.expander("Kaynaklar, hesaplama ve sınırlar"):
        st.markdown(
            "- **Open-Meteo:** Saatlik sıcaklık, bağıl nem, rüzgâr, yön ve yağış; eğitim hattında VPD ve 0–7 cm toprak nemi de kullanılır. Eksik değerler doldurulmaz.\n"
                "- **CFFDRS:** `cffdrs_py` ile FFMC, DMC, DC, ISI, BUI ve birleşik FWI hesaplanır. Sistem bileşenleri önceki kodlara bağlı ardışık günlük kodlardır; kenar çubuğundaki başlangıç değerleri bilinmiyorsa yalnızca açık varsayımlardır.\n"
                "- **Saatlik FFMC:** Depodaki `hourly_fine_fuel_moisture_code` fonksiyonu, kesintisiz saatlik sıcaklık, nem, rüzgâr ve yağış serisiyle ardışık çalıştırılır.\n"
                "- **Yerel geçerlilik:** CFFDRS Kanada için geliştirilmiştir. Diyarbakır anız/otsu yakıt türleri için yerel kalibrasyon yapılmadığından değerler resmî yangın olasılığı veya Türkiye alarm seviyesi değildir.\n"
                "- **DMC/DC mevsim düzeltmesi:** Kütüphane, 30°K üzerindeki enlemlerde 46°K standart gün uzunluğu tablosunu kullanır; Diyarbakır'a özgü değildir.\n"
            "- **NASA FIRMS:** VIIRS S-NPP, NOAA-20, NOAA-21 ve MODIS NRT sorguları yapılır; 4 günlük verinin son 24 saati ekranda gösterilir, önceki günler komşuluk özelliği içindir. Termal nokta yangın teyidi değildir.\n"
            "- **Esri World Imagery + görsel sınıflandırma:** En yüksek öncelikli en çok 8 gerçek nokta paralel incelenir; CORINE 121/131/133 sanayi/maden/inşaat noktaları atlanır. Görselde önce kanıt, sonra arazi/yüzey sınıfı çıkarılır. Görüntünün tarihi olayla aynı olmayabilir; yanık izi veya yangın teyidi değildir.\n"
            "- **Türkiye XGBoost sıralaması:** Model sadece 4 coğrafi fold AUC ortalaması en az 0,65 ise görünür. Görünürse puan, yalnız mevcut haritadaki uygun S-NPP/NOAA-20 noktaları arasındaki yüzdelik sırasıdır; olasılık değildir. Eşik karşılanmıyorsa hiçbir model sırası gösterilmez.\n"
            "- **Model doğrulaması:** Eğitim betiği aynı 1° coğrafi fold'larda persistence özellikli ve persistence özelliksiz modelleri karşılaştırabilir; AUC ortalaması ve standart sapmasını dosyalara yazar. Son dağıtım modeli bütün eğitim satırlarında yeniden kurulur; bu nedenle OOF raporu saha performansı iddiası değildir.\n"
            "- **Emisyon göstergesi:** FRP anlık değer olduğundan 60 saniye varsayımıyla FRE/CO₂/PM₂.₅ hesabı gösterim amaçlıdır; toplam olay emisyonu veya Diyarbakır saha ölçümü değildir.\n"
            "- **Duman konisi ve devriye çizgisi:** Duman konisi rüzgâr yönüne göre basit yön göstergesidir. Devriye sırası en yakın komşu ile kuş uçuşu çizilir; yol ağına göre en kısa rota değildir.\n"
            "- **OpenStreetMap:** Yakın çevre etiketleri aday bağlamdır; yangın nedeni değildir. © OpenStreetMap contributors.\n"
            "- **CORINE Land Cover 2018:** Copernicus/EEA arazi örtüsü sınıfı. Haritalama birimi en az 25 ha olduğundan parsel düzeyinde kanıt değildir; sınıflar 2018 dönemini temsil eder.\n"
            "- **Demo noktaları:** FIRMS anahtarı/bağlantısı yoksa arayüz yedeği olarak otomatik oluşturulur; sentetik olduğu harita, sayaç ve kartta belirtilir. Ekip yönlendirmesinde kullanılmaz.\n"
            "- **FFMC bantları:** <70 düşük, 70–84,9 orta, ≥85 yüksek olarak bu uygulamanın gösterim eşikleridir; resmî Türkiye alarm seviyeleri değildir.\n"
            "- **Haritadaki renkli daire:** Tek hava koordinatının görsel işaretidir; dairenin tümünde aynı risk ölçülmüş değildir.\n"
            "- **Çalışma şekli:** Uygulama arka planda sürekli çalışan bir servis değildir. Veriler 15 dakika önbelleğe alınır; yeni sorgu için yenile düğmesine basın. 7/24 izleme için ayrıca zamanlanmış bir sunucu servisi gerekir."
        )

    st.markdown(
        f"<p class='small-note'>MaydonOs · Son arayüz yenilemesi {datetime.now().strftime('%d.%m.%Y %H:%M')} · "
        "Veri güncelliği bağlı servislerin erişimine bağlıdır.</p>",
        unsafe_allow_html=True,
    )


def main() -> None:
    district, lat, lon, radius, night, firms_key, prev_ffmc, prev_dmc, prev_dc, vision_opt_in = sidebar_inputs()
    st.markdown(
        f"""
        <section class="maydonos-masthead" aria-label="MaydonOs ana başlık">
          <div class="brand-row">
            <span class="brand-name">MaydonOs</span>
            <span class="brand-area">Diyarbakır / {escape(district)}</span>
          </div>
          <p class="brand-kicker">Saha izleme · Güncel görünüm</p>
          <h1>Yangın riski ve uydu bildirimleri</h1>
          <p>Seçilen bölgenin hava koşulları, termal uydu noktaları ve arazi bağlamı. Uydu bildirimi tek başına yangın kanıtı değildir.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='section-intro'>Önce güncel koşulları, ardından haritadaki bildirimleri inceleyin.</p>",
        unsafe_allow_html=True,
    )
    dashboard_content(
        district, lat, lon, radius, night, firms_key,
        prev_ffmc, prev_dmc, prev_dc, vision_opt_in,
    )


if __name__ == "__main__":
    main()
