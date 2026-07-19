"""Baseline credit risk scoring (FR-2 / US-103).

PLACEHOLDER: this is a deterministic, rule-based scorer standing in for a
trained LogisticRegression model until the real Home Credit Default Risk 2018
dataset is sourced (see data/README.md). It is deliberately simple - do not
mistake it for a validated model. Replace score_application() with real
inference once training data is available, but keep the same exclusion rule
below: CODE_GENDER and DAYS_BIRTH must never be used as predictor features,
even in the real model (assumption A-8b, permanent, not just for this stub).
"""

LOW_BAND_MAX = 0.33
MEDIUM_BAND_MAX = 0.66

DTI_SCALE = 0.5
LTI_SCALE = 6.0
TENURE_SCALE_YEARS = 5.0

DTI_WEIGHT = 0.45
LTI_WEIGHT = 0.35
TENURE_WEIGHT = 0.20


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _employment_tenure_years(days_employed) -> float:
    """HC2018 uses a positive sentinel (365243) for unemployed/pensioner
    applicants; treat that (and any non-negative value) as zero tenure."""
    if days_employed is None or days_employed >= 0:
        return 0.0
    return abs(days_employed) / 365.25


def _weighted_contributions(profile: dict) -> dict:
    """Return each risk driver's weighted contribution to the score.

    The single source of truth for the scoring math, shared by score_application
    (which sums it) and explain_score (which labels it). Only reads
    amt_income_total, amt_credit, amt_annuity, days_employed - never
    code_gender or days_birth.
    """
    income = profile.get("amt_income_total")
    credit = profile.get("amt_credit")
    annuity = profile.get("amt_annuity")
    days_employed = profile.get("days_employed")

    if not income or income <= 0:
        raise ValueError("amt_income_total must be a positive number to score an application")
    if credit is None or annuity is None or days_employed is None:
        raise ValueError("amt_credit, amt_annuity, and days_employed are required to score an application")

    dti = annuity / income
    lti = credit / income
    tenure_years = _employment_tenure_years(days_employed)

    return {
        "dti": DTI_WEIGHT * _clamp(dti / DTI_SCALE),
        "lti": LTI_WEIGHT * _clamp(lti / LTI_SCALE),
        "employment_tenure": TENURE_WEIGHT * _clamp(1 - tenure_years / TENURE_SCALE_YEARS),
    }


def score_application(profile: dict) -> tuple[float, str]:
    """Return (probability_of_default, risk_band) for a normalised applicant profile."""
    probability = round(_clamp(sum(_weighted_contributions(profile).values())), 4)

    if probability < LOW_BAND_MAX:
        band = "Low"
    elif probability < MEDIUM_BAND_MAX:
        band = "Medium"
    else:
        band = "High"

    return probability, band


# Human-readable labels for the scorer's three risk drivers.
_FACTOR_LABELS = {
    "dti": "Debt-to-income ratio",
    "lti": "Loan-to-income ratio",
    "employment_tenure": "Employment tenure",
}


def explain_score(profile: dict) -> list[dict]:
    """Return the scorer's top risk factors with weighted contribution + direction.

    A lightweight, always-available stand-in for model SHAP values on the
    rule-based scorer, so the evidence chain (US-307) always has a risk-factor
    component. Contribution = weight * clamped-normalised metric; higher means
    the factor pushed the score toward higher risk. Shares the exact scoring
    math with score_application via _weighted_contributions.
    """
    try:
        contributions = _weighted_contributions(profile)
    except (ValueError, TypeError):
        return []

    factors = [
        {
            "factor": key,
            "label": _FACTOR_LABELS[key],
            "contribution": round(value, 4),
            "direction": "increases_risk" if value > 0 else "decreases_risk",
        }
        for key, value in contributions.items()
    ]
    factors.sort(key=lambda f: f["contribution"], reverse=True)
    return factors
