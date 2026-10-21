"""Create the user's reproducible synthetic demo table (not observational data)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    np.random.seed(42)
    n_samples = 2000

    temp = np.random.uniform(28.0, 46.0, n_samples)
    rh = np.random.uniform(10.0, 50.0, n_samples)
    ws = np.random.uniform(5.0, 45.0, n_samples)
    prec = np.random.choice([0.0, 0.0, 0.0, 0.5], size=n_samples)
    frp = np.random.uniform(2.0, 45.0, n_samples)
    bright_ti4 = np.random.uniform(300.0, 365.0, n_samples)
    landuse_code = np.random.choice([1, 1, 1, 2, 3], size=n_samples)
    is_night = np.random.choice([0, 1], size=n_samples)

    labels: list[int] = []
    for i in range(n_samples):
        is_stubble = (
            (temp[i] > 32.0)
            and (rh[i] < 35.0)
            and (landuse_code[i] == 1)
            and (frp[i] > 7.0)
        )
        if landuse_code[i] == 3:
            labels.append(0)
        else:
            labels.append(1 if is_stubble or (np.random.rand() < 0.08) else 0)

    frame = pd.DataFrame({
        "temp": np.round(temp, 1),
        "rh": np.round(rh, 1),
        "ws": np.round(ws, 1),
        "prec": prec,
        "frp": np.round(frp, 2),
        "bright_ti4": np.round(bright_ti4, 1),
        "landuse_code": landuse_code,
        "is_night": is_night,
        "label": labels,
    })

    out_dir = Path("data/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "turkey_stubble_fire_synthetic_dataset.csv"
    frame.to_csv(csv_path, index=False)
    metadata = {
        "dataset": str(csv_path),
        "synthetic": True,
        "observational_data": False,
        "label_semantics": "synthetic_rule_labels",
        "rows": int(len(frame)),
        "random_seed": 42,
        "label_rule": "landuse_code==3 => 0; otherwise label 1 if temp>32, rh<35, landuse_code==1, frp>7, or random draw<0.08",
        "warning": "Generated for code-path demonstration only. Not NASA FIRMS, field observations, or valid model-performance evidence.",
    }
    csv_path.with_suffix(".meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Synthetic demo dataset created: {csv_path}")
    print(f"Rows: {len(frame):,}")
    print(f"Label counts: {frame['label'].value_counts().sort_index().to_dict()}")
    print("NOTICE: This is synthetic rule-generated data, not real training evidence.")


if __name__ == "__main__":
    main()
