"""Credit risk scoring (FR-2 / US-103, US-201, US-202).

Scores a normalised applicant profile using the trained LightGBM/LogisticRegression
champion model in src/risk_model/ (see reports/ml/ for training metrics and
model_comparison_logreg_vs_lightgbm.md for the selection rationale). Sprint 1's
rule-based placeholder has been replaced, but the permanent constraint it
established stays in force: CODE_GENDER and DAYS_BIRTH must never be used as
predictor features (assumption A-8b). score_application() only ever forwards
the specific keys below to the model - it never reads profile["code_gender"]
or profile["days_birth"], so passing them has no effect on the output.
"""

from src.risk_model.predict import predict_from_profile

# The only profile keys ever forwarded to the model (A-8b: code_gender and
# days_birth are deliberately excluded from this list).
_MODEL_INPUT_KEYS = [
    "amt_income_total",
    "amt_credit",
    "amt_annuity",
    "days_employed",
    "ext_source_1",
    "ext_source_2",
    "ext_source_3",
    "cnt_fam_members",
    "amt_goods_price",
    "cnt_children",
    "name_contract_type",
    "flag_own_car",
    "flag_own_realty",
    "name_income_type",
    "name_education_type",
]


def score_application(profile: dict) -> dict:
    """Score a normalised applicant profile.

    Returns a dict: {"probability": float, "band": str ("Low"/"Medium"/"High"),
    "top_risk_factors": list of {feature, label, value, impact}}.

    Raises ValueError if the minimum required fields for a meaningful score
    (income, credit, annuity, employment tenure) are missing.
    """
    income = profile.get("amt_income_total")
    credit = profile.get("amt_credit")
    annuity = profile.get("amt_annuity")
    days_employed = profile.get("days_employed")

    if not income or income <= 0:
        raise ValueError("amt_income_total must be a positive number to score an application")
    if credit is None or annuity is None or days_employed is None:
        raise ValueError("amt_credit, amt_annuity, and days_employed are required to score an application")

    model_input = {key: profile.get(key) for key in _MODEL_INPUT_KEYS}
    result = predict_from_profile(model_input)

    # HC2018 uses a positive sentinel (365243) for unemployed/pensioner
    # applicants; the training pipeline's clean_data() already maps this to
    # NaN/median for the underlying model, this just keeps the band label
    # naming consistent with Sprint 1 ("Low"/"Medium"/"High" not "LOW"/etc).
    band_label = {"LOW": "Low", "MEDIUM": "Medium", "HIGH": "High"}.get(result["risk_band"], result["risk_band"])

    return {
        "probability": result["risk_score"],
        "band": band_label,
        "top_risk_factors": result["top_risk_factors"],
    }
