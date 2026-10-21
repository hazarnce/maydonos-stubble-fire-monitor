"""Copernicus CORINE 2018 land-cover context for FIRMS detections.

CLC is regional land-cover context, not parcel-scale ground truth. Its status
map minimum mapping unit is 25 ha, so outputs must not be treated as proof of
the cause of a thermal detection.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

CLC_QUERY_URL = (
    "https://copernicus.discomap.eea.europa.eu/arcgis/rest/services/"
    "Corine/CLC2018_LAEA/MapServer/0/query"
)
CLC_TTL_SECONDS = 24 * 60 * 60
MAX_CLC_POINTS = 40
LOCAL_CLC_RASTER = Path(__file__).resolve().parents[1] / "data" / "raw" / "clc2018_raster100m.tif"

# Some Copernicus CLC GeoTIFFs store the ordered 1–44 legend index, not the
# three-digit CLC class. Keep this mapping aligned with the standard CLC legend.
CLC_INDEX_TO_CODE = dict(enumerate([
    111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142,
    211, 212, 213, 221, 222, 223, 231, 241, 242, 243, 244,
    311, 312, 313, 321, 322, 323, 324, 331, 332, 333, 334, 335,
    411, 412, 421, 422, 423, 511, 512, 521, 522, 523,
], start=1))

CLC_NAMES = {
    "111": "Kesintisiz kentsel doku", "112": "Seyrek kentsel doku",
    "121": "Sanayi veya ticaret birimleri", "122": "Yol ve demir yolu alanları",
    "131": "Maden çıkarım alanı", "132": "Depolama alanı", "133": "İnşaat alanı",
    "141": "Kentsel yeşil alan", "142": "Spor ve dinlenme alanı",
    "211": "Sulama yapılmayan ekilebilir arazi", "212": "Sürekli sulanan arazi",
    "213": "Çeltik alanı", "221": "Bağ", "222": "Meyve bahçesi",
    "223": "Zeytinlik", "231": "Mera", "241": "Tek yıllık ve kalıcı ürün karışımı",
    "242": "Karma tarım alanı", "243": "Doğal bitki örtüsü içeren tarım alanı",
    "244": "Tarımsal ormancılık alanı", "311": "Geniş yapraklı orman",
    "312": "İğne yapraklı orman", "313": "Karışık orman",
    "321": "Doğal çayır", "322": "Fundalık", "323": "Maki / çalılık",
    "324": "Geçiş ormanı ve çalılık", "331": "Seyrek bitki örtüsü",
    "332": "Çıplak kaya", "333": "Bitki örtüsü az alan", "334": "Yanmış alan",
    "335": "Buzul ve kalıcı kar", "411": "Bataklık", "412": "Turbalık",
    "421": "Tuzlu bataklık", "422": "Tuzlalar", "423": "Gelgit düzlükleri",
    "511": "Su yolları", "512": "Su kütlesi", "521": "Kıyı lagünü",
    "522": "Haliç", "523": "Deniz",
}


def enrich_hotspots_with_local_raster(
    hotspots: pd.DataFrame,
    raster_path: str | Path = LOCAL_CLC_RASTER,
) -> tuple[pd.DataFrame, str]:
    """Add CLC codes to every point using a local raster, when available.

    Unknown and nodata pixels remain unknown. This bounded, local path is used
    for model inference so the live map and training pipeline can share CLC
    classes without issuing one remote request per hotspot.
    """
    result = hotspots.copy()
    result["corine_code"] = pd.Series([None] * len(result), dtype="object", index=result.index)
    result["corine_class"] = pd.Series([None] * len(result), dtype="object", index=result.index)
    path = Path(raster_path)
    if result.empty:
        return result, "CORINE için harita noktası yok."
    if not path.is_file():
        return result, "Yerel CORINE rasterı yok; model arazi özelliği eşleşen noktalarda hesaplanacak."

    try:
        import numpy as np
        import rasterio
        from rasterio.warp import transform as raster_transform
    except ImportError:
        return result, "rasterio kurulu değil; yerel CORINE örneklemesi kullanılamadı."

    latitudes = pd.to_numeric(result.get("latitude"), errors="coerce")
    longitudes = pd.to_numeric(result.get("longitude"), errors="coerce")
    valid = latitudes.notna() & longitudes.notna()
    if not valid.any():
        return result, "CORINE için geçerli koordinat yok."
    indices = result.index[valid].tolist()
    lats = latitudes.loc[indices].astype(float).tolist()
    lons = longitudes.loc[indices].astype(float).tolist()
    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                return result, "CORINE rasterının CRS bilgisi yok; sınıflar atanmadı."
            xs, ys = raster_transform("EPSG:4326", src.crs, lons, lats)
            found = 0
            for index, sample in zip(indices, src.sample(zip(xs, ys), indexes=1, masked=True)):
                value = sample[0]
                if np.ma.is_masked(value) or not np.isfinite(float(value)):
                    continue
                raw = int(value)
                code = CLC_INDEX_TO_CODE.get(raw)
                if code is None and str(raw) in CLC_NAMES:
                    code = raw
                if code is None:
                    continue
                code_text = str(code)
                result.at[index, "corine_code"] = code_text
                result.at[index, "corine_class"] = CLC_NAMES.get(code_text, f"CLC {code_text}")
                found += 1
    except Exception as exc:
        return result, f"CORINE rasterı okunamadı ({type(exc).__name__}); sınıflar atanmadı."

    return result, f"Copernicus CLC 2018 yerel rasterı · {found}/{len(indices)} nokta eşleşti."


@st.cache_data(ttl=CLC_TTL_SECONDS, max_entries=512, show_spinner=False)
def _lookup_clc_code(lat: float, lon: float) -> tuple[str | None, str | None]:
    """Look up one coordinate using the official Copernicus ArcGIS feature service."""
    params = {
        "geometry": json.dumps({"x": float(lon), "y": float(lat)}),
        "geometryType": "esriGeometryPoint",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "Code_18,ID,Area_Ha",
        "returnGeometry": "false",
        "f": "json",
    }
    response = requests.get(CLC_QUERY_URL, params=params, timeout=(3, 8))
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    if payload.get("error"):
        raise ValueError(str(payload["error"].get("message", "CORINE servisi hata döndürdü")))
    features = payload.get("features") or []
    if not features:
        return None, None
    attrs = features[0].get("attributes") or {}
    raw_code = attrs.get("Code_18")
    code = str(raw_code).strip() if raw_code is not None else None
    return code, CLC_NAMES.get(code or "", "CORINE sınıfı " + code if code else "")


def enrich_hotspots_with_corine(hotspots: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Add CORINE land-cover code/name to real FIRMS rows; never fabricate labels."""
    result = hotspots.copy()
    result["corine_code"] = pd.Series([None] * len(result), dtype="object")
    result["corine_class"] = pd.Series([None] * len(result), dtype="object")
    if result.empty or not {"latitude", "longitude"}.issubset(result.columns):
        return result, "CORINE arazi bağlamı için gerçek sıcak nokta yok."

    real = result
    if "is_demo" in real.columns:
        real = real.loc[~real["is_demo"].fillna(False).astype(bool)]
    if real.empty:
        return result, "Sentetik demo noktalarına CORINE sınıfı atanmadı."

    order = pd.to_numeric(real.get("frp", pd.Series(index=real.index, dtype=float)), errors="coerce")
    ranked_indices = order.sort_values(ascending=False, na_position="last").head(MAX_CLC_POINTS).index
    success = 0
    no_match = 0
    failures = 0
    def lookup(index: Any) -> tuple[Any, str | None, str | None, bool]:
        try:
            code, label = _lookup_clc_code(
                round(float(result.at[index, "latitude"]), 5),
                round(float(result.at[index, "longitude"]), 5),
            )
            return index, code, label, False
        except (requests.RequestException, ValueError, TypeError, KeyError):
            return index, None, None, True

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(lookup, index) for index in ranked_indices]
        for future in as_completed(futures):
            index, code, label, failed = future.result()
            if failed:
                failures += 1
            elif code:
                result.at[index, "corine_code"] = code
                result.at[index, "corine_class"] = label
                success += 1
            else:
                no_match += 1

    if failures and not success:
        status = "Copernicus CORINE servisine erişilemedi; arazi sınıfı belirsiz bırakıldı."
    else:
        status = f"Copernicus CORINE 2018 • {success} nokta eşleşti"
        if no_match:
            status += f" • {no_match} eşleşme yok"
        if failures:
            status += f" • {failures} sorgu başarısız"
    return result, status


def classify_hotspot_tr(row: pd.Series | dict[str, Any], ffmc: float | None) -> tuple[str, str]:
    """Return a cautious Turkish candidate class and marker color.

    Neither FFMC, FRP, OSM, nor 2018 CLC alone verifies illegal burning.
    """
    get = row.get
    if bool(get("is_demo", False)):
        return "Sentetik demo — sınıflandırma yapılmadı", "#ed8b20"

    code_value = get("corine_code")
    code = str(code_value).strip() if code_value is not None and not pd.isna(code_value) else ""
    osm = str(get("osm_candidate", "") or "").casefold()
    frp_value = pd.to_numeric(get("frp"), errors="coerce")
    frp = float(frp_value) if pd.notna(frp_value) else None
    dry = ffmc is not None and pd.notna(ffmc) and float(ffmc) >= 85
    energetic = frp is not None and frp >= 8.0

    industrial_osm = any(word in osm for word in ("sanayi", "enerji", "petrol", "maden"))
    if code.startswith(("121", "131", "132", "133")) or industrial_osm:
        return "Sanayi / tesis çevresi — saha kontrolü gerekli", "#68747b"
    if code.startswith("3"):
        return "Orman / doğal örtüde termal olay adayı", "#a92d2d"
    if code.startswith("2") and dry and energetic:
        return "Tarım alanında yüksek öncelikli sıcak nokta adayı", "#d53b32"
    if dry and energetic:
        return "Kırsal termal olay adayı — arazi sınıfı belirsiz", "#dc8126"
    if code.startswith("2"):
        return "Tarım alanı bağlamı — olay nedeni belirsiz", "#c9a336"
    if code.startswith(("1", "4", "5")):
        return "Diğer arazi bağlamı — olay nedeni belirsiz", "#68747b"
    return "Arazi sınıfı belirsiz — saha incelemesi gerekir", "#68747b"

