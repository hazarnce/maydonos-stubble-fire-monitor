"""Local FIRMS hotspot history and transparent space-time anomaly scoring.

Only real FIRMS records are accepted. Scores are review-priority indicators, not
fire probabilities or validated fire detections.
"""

from __future__ import annotations

import hashlib
import math
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

GRID_SIZE_M = 5_000.0
MIN_HISTORY_DAYS = 7
HISTORY_DAYS = 90
HISTORY_COLUMNS = ["area_id", "date", "cell_x", "cell_y", "count"]


def _area_id(bbox: tuple[float, float, float, float]) -> str:
    normalized = ",".join(f"{float(value):.4f}" for value in bbox)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _xy(latitude: float, longitude: float) -> tuple[float, float]:
    """Approximate local metric grid; adequate for regional aggregation."""
    latitude_factor = math.cos(math.radians(38.0))
    return longitude * 111_320.0 * latitude_factor, latitude * 110_540.0


def _cell(latitude: float, longitude: float) -> tuple[int, int]:
    x, y = _xy(latitude, longitude)
    return math.floor(x / GRID_SIZE_M), math.floor(y / GRID_SIZE_M)


def _bbox_cells(bbox: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    west, south, east, north = map(float, bbox)
    corners = [_cell(south, west), _cell(south, east), _cell(north, west), _cell(north, east)]
    min_x = min(cell[0] for cell in corners)
    max_x = max(cell[0] for cell in corners)
    min_y = min(cell[1] for cell in corners)
    max_y = max(cell[1] for cell in corners)
    return [(x, y) for x in range(min_x, max_x + 1) for y in range(min_y, max_y + 1)]


def _read_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    try:
        frame = pd.read_csv(path)
        if not set(HISTORY_COLUMNS).issubset(frame.columns):
            return pd.DataFrame(columns=HISTORY_COLUMNS)
        frame = frame[HISTORY_COLUMNS].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype("string")
        for column in ("cell_x", "cell_y", "count"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.dropna(subset=HISTORY_COLUMNS).reset_index(drop=True)
    except (OSError, ValueError, pd.errors.ParserError, UnicodeError):
        return pd.DataFrame(columns=HISTORY_COLUMNS)


def record_snapshot(
    bbox: tuple[float, float, float, float],
    hotspots: pd.DataFrame,
    path: str | Path,
    snapshot_date: date | None = None,
) -> pd.DataFrame:
    """Store one daily snapshot for every grid cell covered by this exact area.

    The caller must invoke this only after a successful live FIRMS response;
    synthetic fallback rows are excluded here as an extra guard.
    """
    area = _area_id(bbox)
    day = (snapshot_date or date.today()).isoformat()
    cells = _bbox_cells(bbox)
    counts = {cell: 0 for cell in cells}

    if not hotspots.empty and {"latitude", "longitude"}.issubset(hotspots.columns):
        real = hotspots.copy()
        if "is_demo" in real.columns:
            real = real.loc[~real["is_demo"].fillna(False).astype(bool)]
        for row in real.itertuples(index=False):
            try:
                cell = _cell(float(row.latitude), float(row.longitude))
            except (ValueError, TypeError, AttributeError):
                continue
            if cell in counts:
                counts[cell] += 1

    snapshot = pd.DataFrame([
        {"area_id": area, "date": day, "cell_x": cell[0], "cell_y": cell[1], "count": count}
        for cell, count in counts.items()
    ], columns=HISTORY_COLUMNS)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    history = _read_history(target)
    history = history.loc[~((history["area_id"] == area) & (history["date"] == day))]
    history = pd.concat([history, snapshot], ignore_index=True)
    cutoff = (pd.Timestamp(day) - pd.Timedelta(days=HISTORY_DAYS)).date().isoformat()
    history = history.loc[history["date"] >= cutoff].drop_duplicates(
        subset=["area_id", "date", "cell_x", "cell_y"], keep="last"
    )
    temp_path = target.with_suffix(target.suffix + ".tmp")
    history.to_csv(temp_path, index=False)
    os.replace(temp_path, target)
    return history


def score_snapshot(
    bbox: tuple[float, float, float, float],
    hotspots: pd.DataFrame,
    history: pd.DataFrame,
    snapshot_date: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Attach spatial-temporal review scores and return eligible anomaly cells."""
    points = hotspots.copy()
    points["anomaly_score"] = np.nan
    points["cell_daily_count"] = 0
    points["anomaly_flag"] = False
    if points.empty or "is_demo" not in points.columns:
        return points, pd.DataFrame(), 0

    area = _area_id(bbox)
    day = (snapshot_date or date.today()).isoformat()
    area_history = history.loc[history["area_id"] == area].copy() if not history.empty else pd.DataFrame(columns=HISTORY_COLUMNS)
    if area_history.empty:
        return points, pd.DataFrame(), 0
    current = area_history.loc[area_history["date"] == day]
    prior = area_history.loc[area_history["date"] < day]
    prior_days = int(prior["date"].nunique())
    if prior_days < MIN_HISTORY_DAYS:
        return points, pd.DataFrame(), prior_days

    current_counts = {(int(row.cell_x), int(row.cell_y)): int(row.count) for row in current.itertuples(index=False)}
    historical: dict[tuple[int, int], list[float]] = {}
    for (cell_x, cell_y), group in prior.groupby(["cell_x", "cell_y"]):
        historical[(int(cell_x), int(cell_y))] = group["count"].astype(float).tolist()

    cell_details: list[dict[str, float | int | bool]] = []
    scored_cells: dict[tuple[int, int], tuple[float, int, bool]] = {}
    all_cells = set(current_counts) | set(historical)
    for cell in all_cells:
        baseline = historical.get(cell, [])
        observed = current_counts.get(cell, 0)
        if len(baseline) < MIN_HISTORY_DAYS:
            continue
        median = float(np.median(baseline))
        mad = float(np.median(np.abs(np.asarray(baseline, dtype=float) - median)))
        robust_z = (observed - median) / (1.4826 * mad + 0.5)
        neighbor_excess = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = (cell[0] + dx, cell[1] + dy)
                neighbor_baseline = historical.get(neighbor, [])
                if len(neighbor_baseline) >= MIN_HISTORY_DAYS:
                    if current_counts.get(neighbor, 0) > float(np.median(neighbor_baseline)):
                        neighbor_excess += 1
        score = float(np.clip(50.0 + 10.0 * robust_z + min(neighbor_excess, 4) * 2.5, 0.0, 100.0))
        is_anomaly = observed >= 2 and score >= 70.0 or observed >= 1 and score >= 85.0
        scored_cells[cell] = (score, observed, is_anomaly)
        cell_details.append({
            "grid_x": cell[0], "grid_y": cell[1], "bugünkü_piksel": observed,
            "geçmiş_medyanı": median, "anomali_puanı": round(score, 1),
            "anomali": bool(is_anomaly),
        })

    real_mask = ~points["is_demo"].fillna(False).astype(bool)
    for index, row in points.loc[real_mask].iterrows():
        try:
            cell = _cell(float(row["latitude"]), float(row["longitude"]))
        except (ValueError, TypeError, KeyError):
            continue
        detail = scored_cells.get(cell)
        if detail:
            points.at[index, "anomaly_score"] = detail[0]
            points.at[index, "cell_daily_count"] = detail[1]
            points.at[index, "anomaly_flag"] = detail[2]

    anomaly_cells = pd.DataFrame(cell_details)
    if not anomaly_cells.empty:
        anomaly_cells = anomaly_cells.loc[anomaly_cells["anomali"]].sort_values(
            ["anomali_puanı", "bugünkü_piksel"], ascending=False
        ).reset_index(drop=True)
    return points, anomaly_cells, prior_days
