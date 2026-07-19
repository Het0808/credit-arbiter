"""Proxy leakage assessment (US-305).

Quantifies how strongly the *retained* proxy features OCCUPATION_TYPE and
REGION_RATING_CLIENT are associated with the excluded protected attributes
CODE_GENDER and age (DAYS_BIRTH). Even though the protected attributes are
themselves excluded from the model (A-8b, US-201), a retained feature that is
highly correlated with them can leak indirect discrimination - this report
makes that measurable so the fairness audit can account for it and mitigations
can be planned for Sprint 4.

Association measures:
  - Cramer's V for categorical-vs-categorical (proxy vs CODE_GENDER),
  - correlation ratio (eta) for categorical-vs-continuous (proxy vs age).
Both are in [0, 1]; >= 0.20 is flagged as material leakage here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.risk_model.config import ModelConfig
from src.risk_model.preprocess import load_raw_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports" / "ml"

PROXY_FEATURES = ["OCCUPATION_TYPE", "REGION_RATING_CLIENT"]
MATERIAL_THRESHOLD = 0.20


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Bias-corrected Cramer's V between two categorical series."""
    confusion = pd.crosstab(a, b)
    if confusion.size == 0 or confusion.shape[0] < 2 or confusion.shape[1] < 2:
        return 0.0
    chi2 = _chi2(confusion.to_numpy())
    n = confusion.to_numpy().sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def _chi2(observed: np.ndarray) -> float:
    row = observed.sum(axis=1, keepdims=True)
    col = observed.sum(axis=0, keepdims=True)
    total = observed.sum()
    expected = row @ col / total
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(expected > 0, (observed - expected) ** 2 / expected, 0.0)
    return float(terms.sum())


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Correlation ratio (eta) between a categorical and a continuous variable."""
    df = pd.DataFrame({"cat": categories, "val": values}).dropna()
    if df.empty:
        return 0.0
    grand_mean = df["val"].mean()
    ss_between = sum(
        len(g) * (g["val"].mean() - grand_mean) ** 2 for _, g in df.groupby("cat")
    )
    ss_total = ((df["val"] - grand_mean) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0


def run_proxy_leakage_assessment(df: pd.DataFrame = None) -> dict:
    config = ModelConfig()
    if df is None:
        df = load_raw_data()
    df = df.copy()
    df["_age_years"] = df["DAYS_BIRTH"].abs() / 365.25

    associations = []
    for proxy in PROXY_FEATURES:
        if proxy not in df.columns:
            continue
        v_gender = cramers_v(df[proxy].astype("object"), df["CODE_GENDER"].astype("object"))
        eta_age = correlation_ratio(df[proxy].astype("object"), df["_age_years"])
        associations.append({
            "proxy_feature": proxy,
            "cramers_v_vs_gender": round(v_gender, 4),
            "corr_ratio_vs_age": round(eta_age, 4),
            "material_leakage": bool(max(v_gender, eta_age) >= MATERIAL_THRESHOLD),
        })

    report = {
        "protected_attributes": ["CODE_GENDER", "age (DAYS_BIRTH)"],
        "material_threshold": MATERIAL_THRESHOLD,
        "associations": associations,
        "any_material_leakage": any(a["material_leakage"] for a in associations),
    }
    _write_reports(report)
    return report


def _write_reports(report: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "proxy_leakage.json").write_text(json.dumps(report, indent=2))

    lines = [
        "# Proxy Leakage Assessment (US-305)",
        "",
        "Association between retained proxy features and excluded protected attributes.",
        f"Material-leakage threshold: **{report['material_threshold']}** (Cramer's V / correlation ratio).",
        "",
        "| Proxy feature | Cramer's V vs gender | Corr. ratio vs age | Material leakage |",
        "|---|---|---|---|",
    ]
    for a in report["associations"]:
        lines.append(
            f"| {a['proxy_feature']} | {a['cramers_v_vs_gender']} | {a['corr_ratio_vs_age']} | "
            f"{'YES' if a['material_leakage'] else 'no'} |"
        )
    lines += [
        "",
        "## Mitigations (deferred to Sprint 4)",
        "- For any feature flagged as material leakage, evaluate dropping or transforming it,",
        "  or apply adversarial de-biasing / reweighing before it drives automated decisions.",
        "- Re-run this assessment after any feature-set change and attach it to the fairness audit.",
    ]
    (REPORTS_DIR / "PROXY_LEAKAGE.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    print(json.dumps(run_proxy_leakage_assessment(), indent=2))
