"""Sparse daily FIRMS history for the public web dashboard.

History is an observed-data context, not a fire detector. A coverage marker is
written only after a complete successful FIRMS response, so missing cells can
be interpreted as zero for that covered snapshot without storing every empty
5 km cell across the country.
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
COLUMNS = ["area_id", "date", "cell_x", "cell_y", "count"]
_COVERAGE_CELL = (-1, -1)


def _area_id(bbox: tuple[float, float, float, float]) -> str:
    normalized = ",".join(f"{float(value):.4f}" for value in bbox)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _cell(latitude: float, longitude: float) -> tuple[int, int]:
    x = longitude * 111_320.0 * math.cos(math.radians(38.0))
    y = latitude * 110_540.0
    return math.floor(x / GRID_SIZE_M), math.floor(y / GRID_SIZE_M)


def read_history(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    if not target.exists():
        return pd.DataFrame(columns=COLUMNS)
    try:
        frame = pd.read_csv(target)
        if not set(COLUMNS).issubset(frame.columns):
            return pd.DataFrame(columns=COLUMNS)
        frame = frame[COLUMNS].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for col in ("cell_x", "cell_y", "count"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame.dropna(subset=COLUMNS).reset_index(drop=True)
    except (OSError, ValueError, UnicodeError, pd.errors.ParserError):
        return pd.DataFrame(columns=COLUMNS)


def record_snapshot(
    bbox: tuple[float, float, float, float],
    points: pd.DataFrame,
    path: str | Path,
    snapshot_date: date | None = None,
) -> pd.DataFrame:
    """Record one successfully observed day; empty regions use one marker row."""
    area = _area_id(bbox)
    day = (snapshot_date or date.today()).isoformat()
    west, south, east, north = map(float, bbox)
    counts: dict[tuple[int, int], int] = {}
    if not points.empty and {"latitude", "longitude"}.issubset(points.columns):
        rows = points.copy()
        if "is_demo" in rows:
            rows = rows.loc[~rows["is_demo"].fillna(False).astype(bool)]
        for row in rows.itertuples(index=False):
            try:
                lat, lon = float(row.latitude), float(row.longitude)
                if not (south <= lat <= north and west <= lon <= east):
                    continue
                cell = _cell(lat, lon)
                counts[cell] = counts.get(cell, 0) + 1
            except (ValueError, TypeError, AttributeError):
                continue

    rows = [{"area_id": area, "date": day, "cell_x": _COVERAGE_CELL[0], "cell_y": _COVERAGE_CELL[1], "count": 0}]
    rows.extend({"area_id": area, "date": day, "cell_x": x, "cell_y": y, "count": n} for (x, y), n in counts.items())
    snapshot = pd.DataFrame(rows, columns=COLUMNS)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    history = read_history(target)
    history = history.loc[~((history.area_id == area) & (history.date == day))]
    history = pd.concat([history, snapshot], ignore_index=True)
    cutoff = (pd.Timestamp(day) - pd.Timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    history = history.loc[history.date >= cutoff].drop_duplicates(
        subset=["area_id", "date", "cell_x", "cell_y"], keep="last"
    )
    temporary = target.with_suffix(target.suffix + ".tmp")
    history.to_csv(temporary, index=False)
    os.replace(temporary, target)
    return history


def score_snapshot(
    bbox: tuple[float, float, float, float],
    points: pd.DataFrame,
    history: pd.DataFrame,
    snapshot_date: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Attach a transparent spatial deviation score after seven prior snapshots."""
    scored = points.copy()
    scored["cluster_anomaly_score"] = np.nan
    scored["cluster_daily_count"] = np.nan
    scored["cluster_anomaly"] = False
    area = _area_id(bbox)
    day = (snapshot_date or date.today()).isoformat()
    if scored.empty or history.empty:
        return scored, pd.DataFrame(), 0
    area_history = history.loc[history.area_id == area].copy()
    if area_history.empty:
        return scored, pd.DataFrame(), 0
    prior = area_history.loc[area_history.date < day]
    prior_dates = sorted(prior.date.dropna().unique().tolist())
    prior_days = len(prior_dates)
    if prior_days < MIN_HISTORY_DAYS:
        return scored, pd.DataFrame(), prior_days

    current_counts: dict[tuple[int, int], int] = {}
    for row in area_history.loc[area_history.date == day].itertuples(index=False):
        x, y = int(row.cell_x), int(row.cell_y)
        if (x, y) != _COVERAGE_CELL:
            current_counts[(x, y)] = int(row.count)

    history_counts: dict[tuple[int, int], dict[str, int]] = {}
    for row in prior.itertuples(index=False):
        cell = (int(row.cell_x), int(row.cell_y))
        if cell == _COVERAGE_CELL:
            continue
        history_counts.setdefault(cell, {})[str(row.date)] = int(row.count)

    candidate_cells = set(current_counts) | set(history_counts)
    scored_cells: dict[tuple[int, int], tuple[float, int, bool, float]] = {}
    details: list[dict[str, float | int | bool]] = []
    for cell in candidate_cells:
        records = history_counts.get(cell, {})
        baseline = np.asarray([records.get(prior_day, 0) for prior_day in prior_dates], dtype=float)
        if len(baseline) < MIN_HISTORY_DAYS:
            continue
        median = float(np.median(baseline))
        mad = float(np.median(np.abs(baseline - median)))
        observed = current_counts.get(cell, 0)
        robust_z = (observed - median) / (1.4826 * mad + 0.5)
        neighbor_excess = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                neighbor = (cell[0] + dx, cell[1] + dy)
                neighbor_records = history_counts.get(neighbor, {})
                neighbor_baseline = np.asarray([neighbor_records.get(pday, 0) for pday in prior_dates], dtype=float)
                if len(neighbor_baseline) >= MIN_HISTORY_DAYS and current_counts.get(neighbor, 0) > float(np.median(neighbor_baseline)):
                    neighbor_excess += 1
        score = float(np.clip(50.0 + 10.0 * robust_z + min(neighbor_excess, 4) * 2.5, 0, 100))
        flagged = (observed >= 2 and score >= 70) or (observed >= 1 and score >= 85)
        scored_cells[cell] = (score, observed, flagged, median)
        details.append({"grid_x": cell[0], "grid_y": cell[1], "count_today": observed,
                        "history_median": median, "score": round(score, 1), "is_anomaly": bool(flagged)})

    for index, row in scored.iterrows():
        try:
            cell = _cell(float(row.latitude), float(row.longitude))
        except (ValueError, TypeError, AttributeError):
            continue
        detail = scored_cells.get(cell)
        if detail:
            scored.at[index, "cluster_anomaly_score"] = detail[0]
            scored.at[index, "cluster_daily_count"] = detail[1]
            scored.at[index, "cluster_anomaly"] = detail[2]
    anomalies = pd.DataFrame(details)
    if not anomalies.empty:
        anomalies = anomalies.loc[anomalies.is_anomaly].sort_values(
            ["score", "count_today"], ascending=False
        ).reset_index(drop=True)
    return scored, anomalies, prior_days
