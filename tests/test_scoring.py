import inspect

from src.api.services.scoring import score_application

SAFE_PROFILE = {
    "amt_income_total": 250000,
    "amt_credit": 300000,
    "amt_annuity": 15000,
    "days_employed": -3650,
    "ext_source_1": 0.80,
    "ext_source_2": 0.78,
    "ext_source_3": 0.75,
    "cnt_fam_members": 3,
    "amt_goods_price": 280000,
    "cnt_children": 1,
    "name_contract_type": "Cash loans",
    "flag_own_car": "Y",
    "flag_own_realty": "Y",
    "name_income_type": "Commercial associate",
    "name_education_type": "Higher education",
}

RISKY_PROFILE = {
    "amt_income_total": 80000,
    "amt_credit": 700000,
    "amt_annuity": 45000,
    "days_employed": -90,
    "ext_source_1": 0.10,
    "ext_source_2": 0.12,
    "ext_source_3": 0.08,
    "cnt_fam_members": 3,
    "amt_goods_price": 650000,
    "cnt_children": 1,
    "name_contract_type": "Cash loans",
    "flag_own_car": "N",
    "flag_own_realty": "N",
    "name_income_type": "Working",
    "name_education_type": "Lower secondary",
}


def test_probability_and_band_always_valid():
    for profile in (SAFE_PROFILE, RISKY_PROFILE):
        result = score_application(profile)
        assert 0.0 <= result["probability"] <= 1.0
        assert result["band"] in {"Low", "Medium", "High"}


def test_safe_profile_scores_lower_than_risky_profile():
    """A property test rather than hardcoded thresholds: since the real
    trained model's exact probabilities shift on retraining, we assert
    relative ordering (strong external scores / low DTI-LTI ranks safer)
    rather than pinning exact band cutoffs."""
    safe = score_application(SAFE_PROFILE)
    risky = score_application(RISKY_PROFILE)
    assert safe["probability"] < risky["probability"]


def test_missing_required_fields_raise_value_error():
    incomplete = {"amt_income_total": 100000}
    try:
        score_application(incomplete)
        assert False, "expected ValueError for missing required fields"
    except ValueError:
        pass


def test_days_employed_sentinel_does_not_crash():
    profile = {**SAFE_PROFILE, "days_employed": 365243}
    result = score_application(profile)
    assert 0.0 <= result["probability"] <= 1.0


def test_top_risk_factors_are_well_formed():
    result = score_application(SAFE_PROFILE)
    factors = result["top_risk_factors"]
    assert 1 <= len(factors) <= 5
    for factor in factors:
        assert set(factor.keys()) >= {"feature", "label", "value", "impact"}
        assert isinstance(factor["label"], str) and factor["label"]


def test_scorer_never_reads_protected_attributes_a8b():
    """Regression test for assumption A-8b: CODE_GENDER and DAYS_BIRTH must
    never be read by the risk scorer, even indirectly."""
    source = inspect.getsource(score_application).lower()
    assert '"code_gender"' not in source and "'code_gender'" not in source
    assert '"days_birth"' not in source and "'days_birth'" not in source

    # A profile carrying gender/age must score identically to one without -
    # proving those fields have no effect on the output.
    profile_without = dict(SAFE_PROFILE)
    profile_with = {**profile_without, "code_gender": "F", "days_birth": -9000}
    assert score_application(profile_without) == score_application(profile_with)
