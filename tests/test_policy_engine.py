"""Unit tests for the multi-scheme policy evaluation engine (US-206)."""

from src.api.services import retrieval as R
from src.api.services.policy_engine import compute_metrics, evaluate


def _personal_clauses(query="loan amount income annuity employment"):
    # Retrieve the full personal-loan clause set to feed the evaluator.
    out = R.retrieve("dti income loan-to-income thin file fairness", scheme="Personal Loan", top_k=5)
    return out["clauses"]


def test_compute_metrics_ratios_and_thin_file():
    m = compute_metrics(
        {"amt_income_total": 100000, "amt_credit": 300000, "amt_annuity": 20000, "days_employed": -3650}
    )
    assert round(m["dti"], 2) == 0.20
    assert round(m["lti"], 2) == 3.0
    assert m["thin_file"] is False


def test_employment_sentinel_is_thin_file():
    m = compute_metrics(
        {"amt_income_total": 100000, "amt_credit": 100000, "amt_annuity": 10000, "days_employed": 365243}
    )
    assert m["employment_months"] == 0.0
    assert m["thin_file"] is True


def test_missing_metric_is_unverifiable_and_blocks_approve():
    clauses = _personal_clauses()
    result = evaluate(clauses, {"amt_income_total": None, "amt_credit": 100000, "amt_annuity": 10000})
    assert result["approve_allowed"] is False


def test_compliant_application_passes_all_rules():
    clauses = _personal_clauses()
    profile = {"amt_income_total": 120000, "amt_credit": 200000, "amt_annuity": 20000, "days_employed": -3000}
    result = evaluate(clauses, profile, scheme="Personal Loan")
    assert result["failed_rules"] == []
    assert result["approve_allowed"] is True
    assert result["policy_adherence"] == 1.0


def test_failed_dti_rule_forbids_approve_and_sets_action():
    clauses = _personal_clauses()
    # DTI 0.6 > 0.5 personal-loan cap.
    profile = {"amt_income_total": 100000, "amt_credit": 200000, "amt_annuity": 60000, "days_employed": -3000}
    result = evaluate(clauses, profile, scheme="Personal Loan")
    assert "POL-PL-001" in [f["clause_id"] for f in result["failed_rules"]]
    assert result["approve_allowed"] is False
    assert result["required_action"] in {"refer", "decline"}


def test_thin_file_triggers_escalation():
    clauses = _personal_clauses()
    profile = {"amt_income_total": 120000, "amt_credit": 200000, "amt_annuity": 20000, "days_employed": -300}
    result = evaluate(clauses, profile, scheme="Personal Loan")
    assert result["escalation_required"] is True
    assert "POL-PL-004" in result["escalation_reasons"]
