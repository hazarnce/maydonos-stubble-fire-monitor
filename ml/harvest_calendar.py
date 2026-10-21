"""Province-aware wheat and winter-cereal harvest windows for Turkey.

Windows are broad monitoring periods assembled from Turkish agricultural
research and regional harvest records. They are not parcel-level phenology
observations and should not be presented as exact annual harvest dates.
"""

from __future__ import annotations

import json
import unicodedata
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROVINCE_GEOJSON = ROOT / "frontend" / "assets" / "turkey-boundaries" / "provinces.geojson"

REGIONS: dict[str, str] = {
    "AKD": "Akdeniz",
    "DOG": "Doğu Anadolu",
    "EGE": "Ege",
    "GDA": "Güneydoğu Anadolu",
    "ICA": "İç Anadolu",
    "KAR": "Karadeniz",
    "MAR": "Marmara",
}

PROVINCE_TO_REGION: dict[str, str] = {
    # Akdeniz
    "Adana": "AKD", "Antalya": "AKD", "Burdur": "AKD", "Hatay": "AKD",
    "Isparta": "AKD", "Mersin": "AKD", "Kahramanmaraş": "AKD", "Osmaniye": "AKD",
    # Doğu Anadolu
    "Ağrı": "DOG", "Ardahan": "DOG", "Bingöl": "DOG", "Bitlis": "DOG",
    "Elazığ": "DOG", "Erzincan": "DOG", "Erzurum": "DOG", "Hakkâri": "DOG",
    "Iğdır": "DOG", "Kars": "DOG", "Malatya": "DOG", "Muş": "DOG",
    "Tunceli": "DOG", "Van": "DOG",
    # Ege
    "Afyonkarahisar": "EGE", "Aydın": "EGE", "Denizli": "EGE", "İzmir": "EGE",
    "Kütahya": "EGE", "Manisa": "EGE", "Muğla": "EGE", "Uşak": "EGE",
    # Güneydoğu Anadolu
    "Adıyaman": "GDA", "Batman": "GDA", "Diyarbakır": "GDA", "Gaziantep": "GDA",
    "Mardin": "GDA", "Siirt": "GDA", "Şanlıurfa": "GDA", "Şırnak": "GDA", "Kilis": "GDA",
    # İç Anadolu
    "Aksaray": "ICA", "Ankara": "ICA", "Çankırı": "ICA", "Eskişehir": "ICA",
    "Karaman": "ICA", "Kayseri": "ICA", "Kırıkkale": "ICA", "Kırşehir": "ICA",
    "Konya": "ICA", "Nevşehir": "ICA", "Niğde": "ICA", "Sivas": "ICA", "Yozgat": "ICA",
    # Karadeniz
    "Amasya": "KAR", "Artvin": "KAR", "Bartın": "KAR", "Bayburt": "KAR",
    "Bolu": "KAR", "Çorum": "KAR", "Düzce": "KAR", "Giresun": "KAR",
    "Gümüşhane": "KAR", "Karabük": "KAR", "Kastamonu": "KAR", "Ordu": "KAR",
    "Rize": "KAR", "Samsun": "KAR", "Sinop": "KAR", "Tokat": "KAR",
    "Trabzon": "KAR", "Zonguldak": "KAR",
    # Marmara
    "Balıkesir": "MAR", "Bilecik": "MAR", "Bursa": "MAR", "Çanakkale": "MAR",
    "Edirne": "MAR", "İstanbul": "MAR", "Kırklareli": "MAR", "Kocaeli": "MAR",
    "Sakarya": "MAR", "Tekirdağ": "MAR", "Yalova": "MAR",
}

# (first month/day, last month/day), inclusive. These are cereal-season windows,
# not a claim that every crop or district is harvested on those exact dates.
REGIONAL_HARVEST_WINDOWS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "AKD": ((5, 25), (7, 15)),  # Adana field record: 25 May
    "DOG": ((7, 20), (8, 31)),  # Erzurum field record: 20 July; eastern harvest ends in August
    "EGE": ((5, 20), (7, 15)),  # İzmir: 20 May; inland districts later
    "GDA": ((5, 10), (6, 30)),  # Şanlıurfa: 10 May; Diyarbakır: 30 May
    "ICA": ((5, 25), (7, 25)),  # Aksaray: 25 May; Kayseri: 10 July
    "KAR": ((6, 25), (8, 15)),  # Samsun: 25 June; upland areas later
    "MAR": ((6, 25), (7, 31)),  # Edirne: 25 June; Bandırma: 5 July
}

REGION_FEATURE_NAMES = {
    code: "harvest_region_" + {
        "AKD": "akdeniz", "DOG": "dogu_anadolu", "EGE": "ege",
        "GDA": "guneydogu_anadolu", "ICA": "ic_anadolu", "KAR": "karadeniz",
        "MAR": "marmara",
    }[code]
    for code in REGIONS
}
HARVEST_REGION_FEATURES = [REGION_FEATURE_NAMES[code] for code in REGIONS]
HARVEST_MODEL_FEATURES = ["harvest_progress", *HARVEST_REGION_FEATURES]

HARVEST_SOURCES = [
    {
        "label": "Tarım ve Orman Bakanlığı Tarımsal Araştırmalar · Türkiye buğday hasat sırası",
        "url": "https://arastirma.tarimorman.gov.tr/ttae/Sayfalar/Detay.aspx?SayfaId=47",
    },
    {
        "label": "Toprak Mahsulleri Ofisi · 2016 Hububat Raporu, bölgesel başlangıç tarihleri",
        "url": "https://www.gtb.org.tr/dosya/pdf/2016-yili-hububat-raporu.pdf",
    },
    {
        "label": "Tarım ve Orman Bakanlığı · 2025 bölge bazlı fenolojik gözlemler",
        "url": "https://istatistik.tarimorman.gov.tr/Sayfa/Detay/2089",
    },
    {
        "label": "Tarım ve Orman Bakanlığı · 2026 fenolojik gözlemler",
        "url": "https://istatistik.tarimorman.gov.tr/Sayfa/Detay/2386",
    },
]


def regional_calendar_payload() -> list[dict[str, str]]:
    """Small API-safe summary used by the interface and diagnostics."""
    return [
        {
            "code": code,
            "region": REGIONS[code],
            "start_mm_dd": f"{start[0]:02d}-{start[1]:02d}",
            "end_mm_dd": f"{end[0]:02d}-{end[1]:02d}",
        }
        for code, (start, end) in REGIONAL_HARVEST_WINDOWS.items()
    ]


def _normalize(value: Any) -> str:
    text = str(value or "").strip().casefold().replace("ı", "i")
    return "".join(
        char for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )


_PROVINCE_TO_REGION_NORMALIZED = {
    _normalize(name): code for name, code in PROVINCE_TO_REGION.items()
}


def region_code_for_province(province: Any) -> str | None:
    return _PROVINCE_TO_REGION_NORMALIZED.get(_normalize(province))


def _window_dates(year: int, region_code: str) -> tuple[date, date]:
    (start_month, start_day), (end_month, end_day) = REGIONAL_HARVEST_WINDOWS[region_code]
    return date(year, start_month, start_day), date(year, end_month, end_day)


def harvest_model_features(observation_date: Any, region_code: str) -> dict[str, float]:
    """Return normalized progress and one-hot region inputs for one in-window day."""
    stamp = pd.Timestamp(observation_date)
    start, end = _window_dates(int(stamp.year), region_code)
    day = stamp.date()
    if not start <= day <= end:
        raise ValueError("Gözlem tarihi bu bölgenin tanımlı hububat hasat penceresinde değil.")
    duration = max(1, (end - start).days)
    values = {"harvest_progress": float((day - start).days / duration)}
    values.update({
        REGION_FEATURE_NAMES[code]: float(code == region_code)
        for code in REGIONS
    })
    return values


def is_in_harvest_window(observation_date: Any, region_code: str) -> bool:
    stamp = pd.to_datetime(observation_date, errors="coerce")
    if pd.isna(stamp) or region_code not in REGIONAL_HARVEST_WINDOWS:
        return False
    start, end = _window_dates(int(stamp.year), region_code)
    return start <= stamp.date() <= end


def harvest_context(observation_date: Any, province: Any) -> dict[str, Any]:
    """Describe the province's broad harvest window without inferring fire cause."""
    stamp = pd.to_datetime(observation_date, errors="coerce")
    if pd.isna(stamp):
        return {"label": "Takvim bilgisi yok", "in_period": None, "note": "Gözlem tarihi okunamadı."}
    region_code = region_code_for_province(province)
    if region_code is None:
        return {
            "label": "Bölgesel hasat takvimi bulunamadı", "in_period": None,
            "note": "İl/bölge eşleşmesi yok; takvim uyumu hesaplanmadı.",
        }
    start, end = _window_dates(int(stamp.year), region_code)
    in_period = start <= stamp.date() <= end
    region_name = REGIONS[region_code]
    label = f"{region_name} hububat hasat dönemi" if in_period else f"{region_name} hasat dönemi dışında"
    return {
        "label": label,
        "in_period": bool(in_period),
        "region": region_name,
        "region_code": region_code,
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "note": "Bölgesel hububat takvimi yaklaşık dönem gösterir; sıcak noktanın anız yakımı veya yangın olduğunu kanıtlamaz.",
        "basis": "Kaynaklardaki bölgesel başlangıç örnekleri ve Bakanlığın fenolojik raporlarıyla oluşturulmuş geniş izleme penceresi; yıllık/ilçe bazlı kaymalar olabilir.",
        "sources": HARVEST_SOURCES,
    }


@lru_cache(maxsize=1)
def _province_shapes():
    if not PROVINCE_GEOJSON.is_file():
        raise FileNotFoundError(f"İl sınırları dosyası yok: {PROVINCE_GEOJSON}")
    try:
        from shapely.geometry import shape
        from shapely.strtree import STRtree
    except ImportError as exc:
        raise RuntimeError("Bölgesel takvim için shapely kurulu olmalı.") from exc
    payload = json.loads(PROVINCE_GEOJSON.read_text(encoding="utf-8"))
    features = payload.get("features", [])
    unknown = sorted({
        feature.get("properties", {}).get("name", "")
        for feature in features
        if region_code_for_province(feature.get("properties", {}).get("name")) is None
    })
    if unknown:
        raise ValueError("Hasat takviminde bölgesi tanımlanmamış iller var: " + ", ".join(unknown))
    geometries = tuple(shape(feature["geometry"]) for feature in features)
    return STRtree(geometries), geometries, tuple(features)


@lru_cache(maxsize=250_000)
def _province_at_cached(latitude: float, longitude: float) -> str | None:
    from shapely.geometry import Point

    tree, geometries, features = _province_shapes()
    point = Point(longitude, latitude)
    for candidate in tree.query(point):
        index = int(candidate)
        if geometries[index].covers(point):
            return str(features[index].get("properties", {}).get("name", "")) or None
    return None


def province_at(latitude: Any, longitude: Any) -> str | None:
    try:
        lat, lon = float(latitude), float(longitude)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(lat) or not np.isfinite(lon):
        return None
    return _province_at_cached(round(lat, 5), round(lon, 5))


def annotate_harvest_context(
    frame: pd.DataFrame,
    *,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
    date_column: str = "acq_date",
) -> pd.DataFrame:
    """Add province, region, local window membership, progress and one-hot fields."""
    out = frame.copy()
    if out.empty:
        for column in (
            "province_name", "harvest_region", "harvest_region_code",
            "harvest_start", "harvest_end", "harvest_progress",
            "in_harvest_window", *HARVEST_REGION_FEATURES,
        ):
            out[column] = pd.Series(index=out.index, dtype="object")
        return out
    if not {latitude_column, longitude_column, date_column}.issubset(out.columns):
        raise ValueError("Bölgesel hasat takvimi için enlem, boylam ve tarih sütunları gerekir.")

    latitudes = pd.to_numeric(out[latitude_column], errors="coerce")
    longitudes = pd.to_numeric(out[longitude_column], errors="coerce")
    date_values = pd.to_datetime(out[date_column], errors="coerce")
    coordinate_pairs = pd.DataFrame({"lat": latitudes, "lon": longitudes}).dropna().drop_duplicates()
    province_by_pair = {
        (float(row.lat), float(row.lon)): province_at(row.lat, row.lon)
        for row in coordinate_pairs.itertuples(index=False)
    }
    provinces = [
        province_by_pair.get((float(lat), float(lon))) if pd.notna(lat) and pd.notna(lon) else None
        for lat, lon in zip(latitudes, longitudes)
    ]
    regions = [region_code_for_province(name) for name in provinces]
    years = date_values.dt.year
    doys = date_values.dt.dayofyear
    start_doys = pd.Series(np.nan, index=out.index, dtype=float)
    end_doys = pd.Series(np.nan, index=out.index, dtype=float)
    starts: list[str | None] = [None] * len(out)
    ends: list[str | None] = [None] * len(out)
    progress = pd.Series(np.nan, index=out.index, dtype=float)
    in_window = pd.Series(False, index=out.index, dtype=bool)
    region_series = pd.Series(regions, index=out.index, dtype="object")

    for code in REGIONS:
        region_mask = region_series.eq(code) & years.notna()
        if not region_mask.any():
            continue
        for year in sorted(years.loc[region_mask].dropna().astype(int).unique()):
            mask = region_mask & years.eq(year)
            start, end = _window_dates(int(year), code)
            start_doy = pd.Timestamp(start).dayofyear
            end_doy = pd.Timestamp(end).dayofyear
            start_doys.loc[mask] = start_doy
            end_doys.loc[mask] = end_doy
            selected_indices = np.flatnonzero(mask.to_numpy())
            for index in selected_indices:
                starts[index] = start.isoformat()
                ends[index] = end.isoformat()
            current = (date_values.loc[mask].dt.date >= start) & (date_values.loc[mask].dt.date <= end)
            in_window.loc[mask] = current.to_numpy()
            progress.loc[mask] = ((doys.loc[mask] - start_doy) / max(1, end_doy - start_doy)).to_numpy()

    out["province_name"] = provinces
    out["harvest_region"] = [REGIONS.get(code) if code else None for code in regions]
    out["harvest_region_code"] = regions
    out["harvest_start"] = starts
    out["harvest_end"] = ends
    out["harvest_progress"] = progress
    out["in_harvest_window"] = in_window
    for code, feature in REGION_FEATURE_NAMES.items():
        out[feature] = region_series.eq(code).astype("int8")
    out.attrs["harvest_region_counts"] = {
        REGIONS[code]: int(region_series.eq(code).sum()) for code in REGIONS
    }
    out.attrs["harvest_in_window_counts"] = {
        REGIONS[code]: int((region_series.eq(code) & in_window).sum()) for code in REGIONS
    }
    return out
