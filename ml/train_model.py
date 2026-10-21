"""Train and evaluate a spatially held-out XGBoost FIRMS-event proxy model."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from ml.feature_engineering import (
    MODEL_FEATURES,
    PERSISTENCE_FEATURES,
    calculate_ffmc_features,
    prepare_model_table,
)

MIN_DISPLAY_AUC = 0.65

DEFAULT_DATASET = Path("data/processed/turkey_stubble_fire_dataset.csv")
DEFAULT_MODEL = Path("ml/models/xgboost_turkey_stubble_model.json")


def precision_at_top_k(labels: pd.Series, scores: np.ndarray, ks: tuple[int, ...] = (10, 25, 50)) -> dict[str, Any]:
    """Precision among highest-ranked OOF rows; descriptive for this sampled dataset only."""
    order = np.argsort(-np.asarray(scores, dtype=float))
    y_array = np.asarray(labels, dtype=int)
    result: dict[str, Any] = {}
    for requested_k in ks:
        k = min(requested_k, len(order))
        hits = int(y_array[order[:k]].sum())
        precision = hits / k if k else None
        if k:
            z = 1.96
            denom = 1 + z * z / k
            center = (precision + z * z / (2 * k)) / denom
            half = z * np.sqrt(precision * (1 - precision) / k + z * z / (4 * k * k)) / denom
            interval = [float(max(0, center - half)), float(min(1, center + half))]
        else:
            interval = None
        base_rate = float(y_array.mean()) if len(y_array) else 0.0
        result[str(requested_k)] = {
            "precision": float(precision) if precision is not None else None,
            "positive_proxy_rows_in_top_k": hits,
            "selected_rows": int(k),
            "sample_base_rate": base_rate,
            "lift_vs_sample_base_rate": float(precision / base_rate) if precision is not None and base_rate else None,
            "wilson_95_interval": interval,
        }
    return result


def precision_at_top_fraction(
    labels: pd.Series | np.ndarray,
    scores: np.ndarray,
    fractions: tuple[float, ...] = (0.01, 0.05, 0.10),
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """OOF top-fraction precision/lift with a descriptive Wilson interval."""
    y = np.asarray(labels, dtype=int)
    score_values = np.asarray(scores, dtype=float)
    if len(y) != len(score_values):
        raise ValueError("Precision@fraction için etiket ve skor uzunlukları eşleşmeli.")
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        if len(selected) != len(y):
            raise ValueError("Precision@fraction maskesi satır sayısıyla eşleşmeli.")
        y, score_values = y[selected], score_values[selected]
    if not len(y) or not np.isfinite(score_values).all():
        raise ValueError("Precision@fraction için boş veya sonlu olmayan skor kümesi kullanılamaz.")
    base_rate = float(y.mean())
    order = np.argsort(-score_values)
    result: dict[str, Any] = {}
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("Top-fraction oranı (0, 1] aralığında olmalı.")
        k = max(1, int(round(len(y) * fraction)))
        hits = int(y[order[:k]].sum())
        precision = hits / k
        # Wilson score interval (z=1.96); descriptive only, does not adjust
        # for spatial dependence among observations.
        z = 1.96
        denom = 1 + z * z / k
        center = (precision + z * z / (2 * k)) / denom
        half = z * np.sqrt(precision * (1 - precision) / k + z * z / (4 * k * k)) / denom
        result[f"{fraction:.0%}"] = {
            "k": int(k),
            "positive_proxy_rows": hits,
            "precision": float(precision),
            "lift_vs_sample_base_rate": float(precision / base_rate) if base_rate else None,
            "sample_base_rate": base_rate,
            "wilson_95_interval": [float(max(0, center - half)), float(min(1, center + half))],
        }
    return result


def spatial_groups(frame: pd.DataFrame) -> pd.Series:
    if not {"latitude", "longitude"}.issubset(frame.columns):
        raise ValueError("Rastgele bölme yapılmayacak. Bir derecelik coğrafi gruplar için lat/lon gerekli.")
    lat = pd.to_numeric(frame["latitude"], errors="coerce")
    lon = pd.to_numeric(frame["longitude"], errors="coerce")
    if lat.isna().any() or lon.isna().any() or not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("Coğrafi fold için koordinatlar sonlu ve eksiksiz olmalı.")
    return (
        np.floor(lat).astype(int).astype(str)
        + ":"
        + np.floor(lon).astype(int).astype(str)
    )


def choose_spatial_split(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pick a class-stratified fold while keeping each 1° block in one split."""
    if groups.nunique() < 4:
        raise ValueError("Uzamsal test için en az 4 ayrı 1° coğrafi grup gerekli.")
    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        if y.iloc[train_idx].nunique() == 2 and y.iloc[test_idx].nunique() == 2:
            return train_idx, test_idx, fold
    raise ValueError(
        "Dört uzamsal fold içinde hem pozitif hem negatif içeren test kümesi kurulamadı. "
        "Daha çok bağımsız coğrafi kontrol/olay etiketi ekleyin; rastgele satır bölmesine geçilmedi."
    )


def _load_semantics(dataset: Path, override: str | None) -> tuple[str, dict[str, Any]]:
    meta_path = dataset.with_suffix(".meta.json")
    metadata: dict[str, Any] = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    semantics = override or metadata.get("label_semantics")
    if semantics not in {"firms_hotspot_proxy", "field_verified_illegal_burning"}:
        raise ValueError(
            f"Etiket anlamı tanımsız: {meta_path} içinde label_semantics belirtin veya --label-semantics kullanın."
        )
    return str(semantics), metadata


def train(
    dataset: Path = DEFAULT_DATASET,
    model_path: Path = DEFAULT_MODEL,
    *,
    allow_proxy_labels: bool = False,
    label_semantics_override: str | None = None,
    include_persistence: bool = True,
) -> dict[str, Any]:
    if not dataset.exists():
        raise FileNotFoundError(f"Veri seti yok: {dataset}. Önce `python -m ml.train_dataset_builder` çalıştırın.")
    semantics, source_metadata = _load_semantics(dataset, label_semantics_override)
    if semantics == "firms_hotspot_proxy" and not allow_proxy_labels:
        raise ValueError(
            "Bu veri setinin hedefi kaçak anız yakımı değil, NASA FIRMS sıcak nokta bildirimidir. "
            "Yalnız prototip/araştırma modeli eğitmek için --allow-proxy-labels kullanın; "
            "hukuki/operasyonel sınıflandırma için bağımsız saha teyitli etiket gerekir."
        )
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("XGBoost kurulu değil. `pip install -r requirements-training.txt` çalıştırın.") from exc

    raw = pd.read_csv(dataset, low_memory=False)
    required = {"temp", "rh", "ws", "prec", "landuse_code", "label", "acq_date", "latitude", "longitude"}
    if not required.issubset(raw.columns):
        raise ValueError("CSV'de model girdileriyle birlikte acq_date, latitude ve longitude gerekli.")
    raw = calculate_ffmc_features(raw, ffmc_seed=85.0)
    features = list(MODEL_FEATURES)
    if not include_persistence:
        features = [name for name in features if name not in PERSISTENCE_FEATURES]
    clean, _ = prepare_model_table(raw, features)
    groups = spatial_groups(clean)
    X = clean[features].astype(float)
    y = clean["label"].astype(int)
    if groups.nunique() < 4:
        raise ValueError("4-fold coğrafi doğrulama için en az 4 ayrı spatial group gerekli.")

    splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    oof_probability = np.full(len(clean), np.nan, dtype=float)
    oof_prediction = np.full(len(clean), -1, dtype=int)
    fold_results: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        if y_train.nunique() != 2 or y_test.nunique() != 2:
            raise ValueError(
                f"Coğrafi fold {fold}/4 içinde iki sınıf da yok; güvenilir AUC hesaplanamaz. "
                "Daha fazla mekânsal ve sınıf dengeli doğrulanmış veri ekleyin."
            )
        fold_model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.03,
            scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()),
            objective="binary:logistic", eval_metric="logloss", random_state=42 + fold,
            n_jobs=-1, tree_method="hist", importance_type="gain",
        )
        fold_model.fit(X.iloc[train_idx], y_train, verbose=False)
        probability = fold_model.predict_proba(X.iloc[test_idx])[:, 1]
        prediction = (probability >= 0.5).astype(int)
        auc = float(roc_auc_score(y_test, probability))
        fold_results.append({
            "fold": fold, "auc": auc,
            "macro_f1": float(f1_score(y_test, prediction, average="macro")),
            "test_rows": int(len(test_idx)),
            "held_out_spatial_groups": sorted(groups.iloc[test_idx].unique().tolist()),
        })
        oof_probability[test_idx] = probability
        oof_prediction[test_idx] = prediction

    if not np.isfinite(oof_probability).all() or (oof_prediction < 0).any():
        raise RuntimeError("4-fold değerlendirmede tüm satırlar için OOF tahmini üretilemedi.")
    aucs = np.asarray([item["auc"] for item in fold_results], dtype=float)
    f1s = np.asarray([item["macro_f1"] for item in fold_results], dtype=float)
    report = classification_report(y, oof_prediction, output_dict=True, zero_division=0)
    classwise_auc: dict[str, dict[str, Any]] = {}
    if "corine_code" in clean.columns:
        corine_labels = clean["corine_code"].astype(str).str.replace(r"\.0$", "", regex=True)
        for code in corine_labels.dropna().unique():
            indices = np.flatnonzero(corine_labels.to_numpy() == code)
            class_y = y.iloc[indices]
            positives = int((class_y == 1).sum())
            negatives = int((class_y == 0).sum())
            # Avoid presenting unstable class AUCs from tiny strata.
            if positives >= 5 and negatives >= 5:
                classwise_auc[str(code)] = {
                    "oof_auc": float(roc_auc_score(class_y, oof_probability[indices])),
                    "positive_rows": positives,
                    "negative_rows": negatives,
                    "total_rows": int(len(indices)),
                }
    final_pos_weight = float((y == 0).sum() / (y == 1).sum())
    cold_start_mask = clean["hs_5km_1d"].to_numpy(dtype=float) == 0
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.03,
        scale_pos_weight=final_pos_weight, objective="binary:logistic",
        eval_metric="logloss", random_state=42, n_jobs=-1,
        tree_method="hist", importance_type="gain",
    )
    model.fit(X, y, verbose=False)

    metrics: dict[str, Any] = {
        "label_semantics": semantics,
        "macro_f1_oof": float(f1_score(y, oof_prediction, average="macro")),
        "precision_at_top_k_oof": precision_at_top_k(y, oof_probability),
        "precision_at_top_k_scope": (
            "OOF ranking in the constructed FIRMS-hotspot-proxy/control sample; "
            "not real-world prevalence, calibrated probability, or field-confirmed precision."
        ),
        "evaluation_sample_positive_prevalence": float(y.mean()),
        "precision_at_top_fraction_oof": precision_at_top_fraction(y, oof_probability),
        "precision_at_top_fraction_cold_start_oof": precision_at_top_fraction(
            y, oof_probability, mask=cold_start_mask
        ),
        "cold_start_definition": "hs_5km_1d == 0; no previous-day FIRMS detection within 5 km",
        "cold_start_rows": int(cold_start_mask.sum()),
        "cold_start_positive_proxy_rows": int(y.to_numpy()[cold_start_mask].sum()),
        "precision_interpretation_limit": (
            "Ranking metrics and Wilson intervals describe this sampled FIRMS-proxy/control dataset only; "
            "intervals do not adjust for spatial dependence and are not real-prevalence performance."
        ),
        "roc_auc_oof": float(roc_auc_score(y, oof_probability)),
        "spatial_cv_auc_mean": float(aucs.mean()),
        "spatial_cv_auc_std": float(aucs.std(ddof=1)),
        "spatial_cv_auc_min": float(aucs.min()),
        "spatial_cv_auc_max": float(aucs.max()),
        "spatial_cv_auc_range": [float(aucs.min()), float(aucs.max())],
        "spatial_cv_macro_f1_mean": float(f1s.mean()),
        "spatial_cv_macro_f1_std": float(f1s.std(ddof=1)),
        "spatial_cv_folds": fold_results,
        "classification_report_oof": report,
        "corine_classwise_oof_auc": classwise_auc,
        "corine_classwise_auc_minimum_support_each_class": 5,
        "corine_classwise_auc_interpretation": (
            "Exploratory direction only; classwise estimates may be noisy. Use the grain profile as the primary result."
        ),
        "test_rows_total": int(len(clean)),
        "spatial_group_count": int(groups.nunique()),
        "features": features,
        "model_variant": "with_persistence" if include_persistence else "without_persistence",
        "minimum_display_auc": MIN_DISPLAY_AUC,
        "excluded_features_to_prevent_leakage": ["frp", "bright_ti4", "is_night", "label_basis"],
        "scale_pos_weight_full_fit": final_pos_weight,
        "threshold": 0.5,
        "model_parameters": {
            "n_estimators": 300, "max_depth": 6, "learning_rate": 0.03,
            "objective": "binary:logistic", "random_state": 42,
        },
        "target_limit": (
            "Model predicts FIRMS hotspot proxy vs selected controls; it cannot determine fire legality or cause."
            if semantics == "firms_hotspot_proxy"
            else "Target follows the independent field-verified labels described by the dataset owner."
        ),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset),
        "dataset_rows": int(len(clean)),
        "source_dataset_metadata": source_metadata,
        "deployment_note": "The saved production model is refit on all rows after grouped out-of-fold evaluation.",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    model_path.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    importances = pd.DataFrame(
        {"feature": features, "importance_gain": model.feature_importances_}
    ).sort_values("importance_gain", ascending=False)
    importances.to_csv(model_path.with_suffix(".feature_importance.csv"), index=False)
    print(f"--- TÜRKİYE XGBOOST · 4-FOLD 1° COĞRAFİ DEĞERLENDİRME · {metrics['model_variant']} ---")
    print(f"Etiket: {semantics}; tüm satırlar grouped out-of-fold değerlendirildi")
    for item in fold_results:
        print(f"Fold {item['fold']}: AUC={item['auc']:.4f}, Macro F1={item['macro_f1']:.4f}")
    print(
        f"AUC 4-fold aralığı: {metrics['spatial_cv_auc_min']:.4f}–{metrics['spatial_cv_auc_max']:.4f}; "
        f"ortalama ± std: {metrics['spatial_cv_auc_mean']:.4f} ± {metrics['spatial_cv_auc_std']:.4f}"
    )
    print(f"Macro F1 OOF: {metrics['macro_f1_oof']:.4f}")
    print("OOF Precision@K (sample-only, not operational prevalence): " + json.dumps(
        metrics["precision_at_top_k_oof"], ensure_ascii=False
    ))
    print("OOF top-fraction precision/lift: " + json.dumps(
        metrics["precision_at_top_fraction_oof"], ensure_ascii=False
    ))
    print("OOF cold-start top-fraction precision/lift: " + json.dumps(
        metrics["precision_at_top_fraction_cold_start_oof"], ensure_ascii=False
    ))
    print(classification_report(y, oof_prediction, zero_division=0))
    print(f"Model: {model_path}")
    print("Not: FIRMS proxy etiketiyle çıkan metrikler kaçak anız yakımı başarısı değildir.")
    return metrics


def train_ablation(
    dataset: Path = DEFAULT_DATASET,
    model_path: Path = DEFAULT_MODEL,
    *,
    allow_proxy_labels: bool = False,
    label_semantics_override: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Fit persistence/no-persistence variants on identical rows and fixed folds."""
    without_path = model_path.with_name(f"{model_path.stem}_without_persistence{model_path.suffix}")
    with_result = train(
        dataset, model_path, allow_proxy_labels=allow_proxy_labels,
        label_semantics_override=label_semantics_override, include_persistence=True,
    )
    without_result = train(
        dataset, without_path, allow_proxy_labels=allow_proxy_labels,
        label_semantics_override=label_semantics_override, include_persistence=False,
    )
    comparison = {
        "dataset": str(dataset),
        "label_semantics": with_result["label_semantics"],
        "spatial_validation": "4-fold StratifiedGroupKFold; fixed 1-degree geographic groups; identical rows and folds",
        "with_persistence": {
            "auc_mean": with_result["spatial_cv_auc_mean"],
            "auc_std": with_result["spatial_cv_auc_std"],
            "auc_range": [with_result["spatial_cv_auc_min"], with_result["spatial_cv_auc_max"]],
            "fold_auc": [item["auc"] for item in with_result["spatial_cv_folds"]],
            "oof_macro_f1": with_result["macro_f1_oof"],
            "precision_at_top_k_oof": with_result["precision_at_top_k_oof"],
            "precision_at_top_fraction_oof": with_result["precision_at_top_fraction_oof"],
            "precision_at_top_fraction_cold_start_oof": with_result["precision_at_top_fraction_cold_start_oof"],
        },
        "without_persistence": {
            "auc_mean": without_result["spatial_cv_auc_mean"],
            "auc_std": without_result["spatial_cv_auc_std"],
            "auc_range": [without_result["spatial_cv_auc_min"], without_result["spatial_cv_auc_max"]],
            "fold_auc": [item["auc"] for item in without_result["spatial_cv_folds"]],
            "oof_macro_f1": without_result["macro_f1_oof"],
            "precision_at_top_k_oof": without_result["precision_at_top_k_oof"],
            "precision_at_top_fraction_oof": without_result["precision_at_top_fraction_oof"],
            "precision_at_top_fraction_cold_start_oof": without_result["precision_at_top_fraction_cold_start_oof"],
        },
        "auc_delta_with_minus_without": (
            with_result["spatial_cv_auc_mean"] - without_result["spatial_cv_auc_mean"]
        ),
        "interpretation_limit": (
            "FIRMS thermal-hotspot proxy versus selected no-detection controls; not illegal burning, "
            "fire ignition, or field-confirmed performance."
        ),
    }
    comparison_path = model_path.with_suffix(".ablation.metrics.json")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print("--- PERSISTENCE ABLASYON KARŞILAŞTIRMASI ---")
    print(
        "Persistence var: "
        f"AUC {with_result['spatial_cv_auc_min']:.4f}–{with_result['spatial_cv_auc_max']:.4f} "
        f"(ortalama {with_result['spatial_cv_auc_mean']:.4f} ± {with_result['spatial_cv_auc_std']:.4f})"
    )
    print(
        "Persistence yok: "
        f"AUC {without_result['spatial_cv_auc_min']:.4f}–{without_result['spatial_cv_auc_max']:.4f} "
        f"(ortalama {without_result['spatial_cv_auc_mean']:.4f} ± {without_result['spatial_cv_auc_std']:.4f})"
    )
    print(
        "Fark (var − yok): "
        f"{with_result['spatial_cv_auc_mean'] - without_result['spatial_cv_auc_mean']:+.4f} AUC"
    )
    print("Bu karşılaştırma FIRMS sıcak nokta vekil etiketini ölçer; yangın veya tutuşturma tahmini değildir.")
    print(f"Ablasyon raporu: {comparison_path}")
    return {"with_persistence": with_result, "without_persistence": without_result}

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Türkiye uzamsal XGBoost model eğitimi")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--allow-proxy-labels", action="store_true", help="FIRMS tespiti vekil etiketleriyle araştırma prototipi eğitmeye açıkça izin verir.")
    parser.add_argument("--label-semantics", choices=["firms_hotspot_proxy", "field_verified_illegal_burning"], default=None)
    parser.add_argument("--without-persistence", action="store_true", help="hs_* özelliklerini dışarıda bırakarak ablation kontrol modelini eğitir.")
    parser.add_argument("--ablation", action="store_true", help="Aynı veri ve coğrafi fold'larda persistence var/yok iki modeli eğitir.")
    args = parser.parse_args()
    if args.ablation and args.without_persistence:
        parser.error("--ablation ve --without-persistence birlikte kullanılamaz.")
    if args.ablation:
        train_ablation(args.dataset, args.model_out, allow_proxy_labels=args.allow_proxy_labels, label_semantics_override=args.label_semantics)
    else:
        train(
            args.dataset, args.model_out,
            allow_proxy_labels=args.allow_proxy_labels,
            label_semantics_override=args.label_semantics,
            include_persistence=not args.without_persistence,
        )


if __name__ == "__main__":
    main()
