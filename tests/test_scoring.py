import inspect

from src.api.services.scoring import score_application


def test_low_dti_low_lti_long_tenure_is_low_band():
    profile = {
        "amt_income_total": 250000,
        "amt_credit": 300000,
        "amt_annuity": 15000,
        "days_employed": -3650,
    }
    probability, band = score_application(profile)
    assert 0.0 <= probability <= 1.0
    assert band == "Low"


def test_high_dti_high_lti_short_tenure_is_high_band():
    profile = {
        "amt_income_total": 80000,
        "amt_credit": 700000,
        "amt_annuity": 45000,
        "days_employed": -90,
    }
    probability, band = score_application(profile)
    assert 0.0 <= probability <= 1.0
    assert band == "High"


def test_probability_always_in_unit_interval_across_extreme_inputs():
    extreme_profiles = [
        {"amt_income_total": 1, "amt_credit": 10_000_000, "amt_annuity": 5_000_000, "days_employed": -1},
        {"amt_income_total": 1_000_000, "amt_credit": 0, "amt_annuity": 0, "days_employed": -20000},
        {"amt_income_total": 50000, "amt_credit": 100000, "amt_annuity": 10000, "days_employed": 365243},
    ]
    for profile in extreme_profiles:
        probability, band = score_application(profile)
        assert 0.0 <= probability <= 1.0
        assert band in {"Low", "Medium", "High"}


def test_days_employed_sentinel_treated_as_zero_tenure():
    with_sentinel = {"amt_income_total": 140000, "amt_credit": 350000, "amt_annuity": 18000, "days_employed": 365243}
    with_zero_tenure = {"amt_income_total": 140000, "amt_credit": 350000, "amt_annuity": 18000, "days_employed": 0}
    assert score_application(with_sentinel) == score_application(with_zero_tenure)


def test_scorer_never_reads_protected_attributes_a8b():
    """Regression test for assumption A-8b: CODE_GENDER and DAYS_BIRTH must
    never be read by the risk scorer, even indirectly."""
    source = inspect.getsource(score_application).lower()
    # Search for actual dict-key access (quoted), not prose mentions in the docstring.
    assert '"code_gender"' not in source and "'code_gender'" not in source
    assert '"days_birth"' not in source and "'days_birth'" not in source

    # A profile missing gender/age entirely must score identically to one that
    # has them set - proving the fields have no effect on the output.
    profile_without = {"amt_income_total": 120000, "amt_credit": 400000, "amt_annuity": 20000, "days_employed": -900}
    profile_with = {**profile_without, "code_gender": "F", "days_birth": -9000}
    assert score_application(profile_without) == score_application(profile_with)
