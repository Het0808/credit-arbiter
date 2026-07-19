from src.api.services.policy_evaluation import evaluate_policy


def test_clean_personal_profile_passes_all_rules():
    profile = {"amt_income_total": 250000, "amt_credit": 300000, "amt_annuity": 15000, "days_employed": -3650}
    result = evaluate_policy(profile, "Personal")
    assert result["failed_rules"] == []
    assert len(result["passed_rules"]) == 3
    assert result["thin_file_flag"] is False
    assert result["escalation_required"] is False


def test_high_dti_fails_dti_rule():
    profile = {"amt_income_total": 80000, "amt_credit": 200000, "amt_annuity": 45000, "days_employed": -3650}
    result = evaluate_policy(profile, "Personal")
    failed_rule_names = {r["rule"] for r in result["failed_rules"]}
    assert "max_dti" in failed_rule_names
    assert result["escalation_required"] is True


def test_unemployment_sentinel_sets_thin_file_flag():
    profile = {"amt_income_total": 140000, "amt_credit": 350000, "amt_annuity": 18000, "days_employed": 365243}
    result = evaluate_policy(profile, "Personal")
    assert result["thin_file_flag"] is True
    assert result["escalation_required"] is True


def test_first_time_borrower_is_always_thin_file():
    profile = {"amt_income_total": 50000, "amt_credit": 100000, "amt_annuity": 5000, "days_employed": -1000}
    result = evaluate_policy(profile, "First-Time Borrower")
    assert result["thin_file_flag"] is True
    assert result["escalation_required"] is True


def test_low_income_assistance_uses_reduced_income_floor():
    profile = {"amt_income_total": 9000, "amt_credit": 20000, "amt_annuity": 1000, "days_employed": -1000}
    result = evaluate_policy(profile, "Low-Income Assistance")
    failed_rule_names = {r["rule"] for r in result["failed_rules"]}
    assert "min_income" not in failed_rule_names  # 9000 clears the 8000 floor for this scheme


def test_unknown_scheme_falls_back_to_personal_rules():
    profile = {"amt_income_total": 250000, "amt_credit": 300000, "amt_annuity": 15000, "days_employed": -3650}
    result = evaluate_policy(profile, None)
    assert result["scheme"] == "Personal"
