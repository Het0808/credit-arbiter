"""Production-grade hardened ML training (US-201).

Trains the champion LightGBM risk model on the real Home Credit data with:
  - the fairness-clean feature set (CODE_GENDER / DAYS_BIRTH / AGE_YEARS excluded),
  - auxiliary bureau + previous_application aggregate features (aux_features.py),
  - stronger, regularised hyperparameters,
and evaluates AUC + threshold-optimised F1 on a held-out test set. On success it
overwrites the production model (models/production/risk_model_v1.pkl) and metadata.

Run:  python -m src.risk_model.train_hardened
"""

from datetime import date
import json
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.risk_model.aux_features import build_aux_features, merge_aux_features
from src.risk_model.config import ModelConfig
from src.risk_model.preprocess import get_preprocessor, load_raw_data, prepare_pipeline_data, split_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROD_DIR = PROJECT_ROOT / "models" / "production"
PROD_MODEL_PATH = PROD_DIR / "risk_model_v1.pkl"
PROD_METADATA_PATH = PROD_DIR / "model_metadata.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "ml" / "hardened_metrics.json"

# Regularised LightGBM params tuned for Home Credit's ~8% positive rate.
HARDENED_PARAMS = dict(
    n_estimators=700,
    learning_rate=0.02,
    num_leaves=34,
    max_depth=8,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.5,
    min_child_samples=80,
    n_jobs=-1,
    verbose=-1,
)


def _best_f1_threshold(y_true, y_prob):
    """Return (threshold, f1) maximising F1 over a grid - the imbalance-aware
    operating point, since a 0.5 cut-off is meaningless at 8% prevalence."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def train():
    config = ModelConfig()
    print("Loading application data + auxiliary aggregates...")
    df = load_raw_data()
    aux = build_aux_features()
    df = merge_aux_features(df, aux)

    X, y, _ = prepare_pipeline_data(df, config)
    print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} cols")

    # Fairness assertion (US-201 AC): protected attrs must be absent from features.
    for banned in config.PROTECTED_EXCLUDED_FEATURES:
        assert banned not in X.columns, f"Protected feature {banned} leaked into X!"

    X_train, X_test, y_train, y_test = split_data(X, y, config)
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    pipeline = Pipeline([
        ("preprocessor", get_preprocessor(config)),
        ("classifier", LGBMClassifier(random_state=config.RANDOM_STATE,
                                      scale_pos_weight=scale_pos_weight, **HARDENED_PARAMS)),
    ])

    print("Training hardened LightGBM...")
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, y_prob))
    best_t, best_f1 = _best_f1_threshold(y_test.values, y_prob)
    precision = float(precision_score(y_test, (y_prob >= best_t).astype(int), zero_division=0))
    recall = float(recall_score(y_test, (y_prob >= best_t).astype(int), zero_division=0))

    metrics = {
        "roc_auc": round(auc, 4),
        "best_f1": round(best_f1, 4),
        "f1_threshold": round(best_t, 3),
        "precision_at_best_f1": round(precision, 4),
        "recall_at_best_f1": round(recall, 4),
        "feature_count": int(X.shape[1]),
        "aux_features_used": config.AUX_FEATURES,
        "protected_features_excluded": config.PROTECTED_EXCLUDED_FEATURES,
        "auc_target_met": auc >= 0.80,
        "training_date": date.today().isoformat(),
    }

    print("\n=== HARDENED MODEL ===")
    print(f"ROC-AUC : {auc:.4f}  (target >= 0.80 -> {'MET' if auc >= 0.80 else 'NOT MET'})")
    print(f"Best F1 : {best_f1:.4f} @ threshold {best_t:.2f}  (P={precision:.3f} R={recall:.3f})")
    print(f"Features: {X.shape[1]} (protected excluded: {config.PROTECTED_EXCLUDED_FEATURES})")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(metrics, indent=2))

    PROD_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PROD_MODEL_PATH)

    metadata = {
        "selected_model_name": "LightGBM (Hardened, Aux Features)",
        "selected_model_version": "v1_hardened",
        "reason_for_selection": "US-201 hardening: aux bureau/previous-application aggregates + "
                                "fairness-clean feature set (CODE_GENDER/DAYS_BIRTH/AGE_YEARS excluded).",
        "ROC-AUC": auc,
        "Best-F1": best_f1,
        "F1-threshold": best_t,
        "Precision": precision,
        "Recall": recall,
        "feature_count": int(X.shape[1]),
        "protected_features_excluded": config.PROTECTED_EXCLUDED_FEATURES,
        "threshold_policy": {"conservative": 0.36, "balanced": round(best_t, 2), "revenue_friendly": 0.75},
        "training_date": date.today().isoformat(),
    }
    PROD_METADATA_PATH.write_text(json.dumps(metadata, indent=4))
    print(f"\nSaved production model -> {PROD_MODEL_PATH}")
    print(f"Saved metadata         -> {PROD_METADATA_PATH}")
    return metrics


if __name__ == "__main__":
    train()
