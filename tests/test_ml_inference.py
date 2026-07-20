"""Tests for the trained-model inference bridge on NEW applications (US-201 serving).

Verifies that the trained LightGBM model scores brand-new applications (not just
historical dataset IDs), and that the assessment flow uses it when RISK_SCORER=ml
and gracefully falls back to the rule-based scorer otherwise.
"""

import json

import pytest

from src.api.models import Application
from src.api.services import assessment as assessment_module
from src.api.services.assessment import run_assessment

NEW_APPLICANT = {
    "SK_ID_CURR": "900001", "AMT_INCOME_TOTAL": 180000, "AMT_CREDIT": 450000,
    "AMT_ANNUITY": 24000, "AMT_GOODS_PRICE": 420000, "DAYS_EMPLOYED": -2200,
    "DAYS_BIRTH": -13000, "CNT_FAM_MEMBERS": 2, "CNT_CHILDREN": 0,
    "EXT_SOURCE_1": 0.6, "EXT_SOURCE_2": 0.55, "EXT_SOURCE_3": 0.5,
    "NAME_CONTRACT_TYPE": "Cash loans", "NAME_INCOME_TYPE": "Working",
    "NAME_EDUCATION_TYPE": "Higher education", "FLAG_OWN_CAR": "Y", "FLAG_OWN_REALTY": "Y",
}


def _app_from(raw: dict, db):
    app = Application(
        external_id=str(raw.get("SK_ID_CURR", "900001")),
        loan_scheme="Personal Loan",
        amt_income_total=float(raw["AMT_INCOME_TOTAL"]),
        amt_credit=float(raw["AMT_CREDIT"]),
        amt_annuity=float(raw["AMT_ANNUITY"]),
        days_employed=int(raw["DAYS_EMPLOYED"]),
        status="COMPLETE",
        raw_row_json=json.dumps(raw),
    )
    db.add(app); db.commit(); db.refresh(app)
    return app


def test_predict_scores_a_brand_new_applicant():
    predict = pytest.importorskip("src.risk_model.predict")
    result = predict.predict_from_features(NEW_APPLICANT)
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["risk_band"] in {"LOW", "MEDIUM", "HIGH"}
    assert "LightGBM" in result["model_type"]


def test_predict_tolerates_sparse_input():
    predict = pytest.importorskip("src.risk_model.predict")
    sparse = {"SK_ID_CURR": "900002", "AMT_INCOME_TOTAL": 90000, "AMT_CREDIT": 500000,
              "AMT_ANNUITY": 35000, "DAYS_EMPLOYED": -300}
    result = predict.predict_from_features(sparse)
    assert 0.0 <= result["risk_score"] <= 1.0


def test_assessment_uses_ml_model_when_enabled(db_session, monkeypatch):
    pytest.importorskip("lightgbm")
    monkeypatch.setenv("RISK_SCORER", "ml")
    record = run_assessment(db_session, _app_from(NEW_APPLICANT, db_session))
    evidence = json.loads(record.evidence_chain_json)
    assert evidence["scorer"] == "ml_model"
    assert 0.0 <= record.risk_score <= 1.0
    assert record.risk_band in {"Low", "Medium", "High"}


def test_assessment_defaults_to_rule_based(db_session):
    # No RISK_SCORER env -> deterministic rule-based scorer.
    record = run_assessment(db_session, _app_from(NEW_APPLICANT, db_session))
    evidence = json.loads(record.evidence_chain_json)
    assert evidence["scorer"] == "rule_based"


def test_ml_falls_back_when_no_raw_fields(db_session, monkeypatch):
    monkeypatch.setenv("RISK_SCORER", "ml")
    app = Application(external_id="900003", loan_scheme="Personal Loan", amt_income_total=250000,
                      amt_credit=300000, amt_annuity=15000, days_employed=-3650, status="COMPLETE")
    db_session.add(app); db_session.commit(); db_session.refresh(app)
    record = run_assessment(db_session, app)  # raw_row_json is None -> fall back
    evidence = json.loads(record.evidence_chain_json)
    assert evidence["scorer"] == "rule_based"
