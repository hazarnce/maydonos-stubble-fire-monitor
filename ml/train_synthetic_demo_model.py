"""Train a clearly marked XGBoost demo model on synthetic rule labels.

This script is deliberately separate from train_model.py. Its output is not a
production/field-validated fire classifier and must not be used operationally.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from sklearn.metrics import classification_report, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
except ImportError as exc:
    raise RuntimeError(
        "scikit-learn kurulu değil. Ağ bağlantısı olan ortamda "
        "`python -m pip install scikit-learn` çalıştırın."
    ) from exc

from ml.feature_engineering import MODEL_FEATURES, calculate_ffmc_features, prepare_model_table


def main() -> None:
    dataset = Path("data/synthetic/turkey_stubble_fire_synthetic_dataset.csv")
    output_dir = Path("ml/models/synthetic_demo")
    model_path = output_dir / "xgboost_synthetic_demo_model.json"
    metrics_path = output_dir / "xgboost_synthetic_demo_model.metrics.json"

    if not dataset.is_file():
        raise FileNotFoundError(f"Sentetik demo veri seti bulunamadı: {dataset}")

    raw = pd.read_csv(dataset)
    raw = calculate_ffmc_features(raw, ffmc_seed=85.0)
    clean, pos_weight = prepare_model_table(raw)
    X = clean[MODEL_FEATURES].astype(float)
    y = clean["label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost kurulu değil. Ağ bağlantısı olan ortamda "
            "`python -m pip install xgboost` çalıştırın."
        ) from exc

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        scale_pos_weight=pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        importance_type="gain",
    )
    model.fit(X_train, y_train, verbose=False)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    metrics = {
        "model_type": "synthetic_demo_only",
        "dataset": str(dataset),
        "dataset_semantics": "synthetic_rule_labels",
        "observational_data": False,
        "operational_use_allowed": False,
        "split_method": "random stratified 75/25; spatial validation unavailable because synthetic data have no coordinates",
        "rows": int(len(clean)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "class_counts": {str(int(k)): int(v) for k, v in y.value_counts().sort_index().items()},
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "classification_report": report,
        "features": MODEL_FEATURES,
        "features_not_used": ["frp", "bright_ti4", "is_night"],
        "threshold": 0.5,
        "random_seed": 42,
        "scale_pos_weight": float(pos_weight),
        "model_parameters": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.03,
            "objective": "binary:logistic",
            "random_state": 42,
        },
        "limitation": "Metrics measure reproduction of synthetic rule-generated labels only; they are not evidence of real fire or illegal stubble-burning detection accuracy.",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (
        pd.DataFrame({"feature": MODEL_FEATURES, "importance_gain": model.feature_importances_})
        .sort_values("importance_gain", ascending=False)
        .to_csv(output_dir / "xgboost_synthetic_demo_feature_importance.csv", index=False)
    )

    print("--- SENTETİK DEMO XGBOOST EĞİTİMİ ---")
    print(f"Etiket dağılımı: {metrics['class_counts']}")
    print(f"Eğitim/test satırı: {len(X_train)}/{len(X_test)}")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Model: {model_path}")
    print(f"Metrikler: {metrics_path}")
    print("UYARI: Bu skorlar sentetik kuralları öğrenme başarısıdır; gerçek saha başarısı değildir.")


if __name__ == "__main__":
    main()
