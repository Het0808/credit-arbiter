"""Tests for escalation workflow + complete evidence chain (US-306, US-307)."""

import json

from src.api.models import Application, DecisionRecord, SchemePause
from src.api.services import assessment as assessment_module
from src.api.services.assessment import run_assessment


def _complete_app(db, scheme="Personal Loan"):
    app = Application(
        external_id="ev-1", loan_scheme=scheme, amt_income_total=250000, amt_credit=300000,
        amt_annuity=15000, days_employed=-3650, days_birth=-14200, code_gender="F", status="COMPLETE",
    )
    db.add(app)
    db.commit()
    db.refresh(app)
    return app


def test_evidence_chain_has_all_six_components(db_session):
    record = run_assessment(db_session, _complete_app(db_session))
    evidence = json.loads(record.evidence_chain_json)
    present = evidence["evidence_present"]
    for component in ("risk_score", "risk_factors", "policy_clauses", "document_findings",
                      "regulatory_result", "fairness_result"):
        assert component in present
    assert record.evidence_complete is True
    assert evidence["risk_factors"], "risk factors must be populated"


def test_low_confidence_forces_refer(db_session, monkeypatch):
    # Score near 0.5 -> confidence ~0.5 < 0.60 -> forced Refer (US-306).
    monkeypatch.setattr(assessment_module, "score_application", lambda profile: (0.52, "Medium"))
    record = run_assessment(db_session, _complete_app(db_session))
    assert record.recommendation == "Refer"
    assert record.escalation_reason_code == "low_model_confidence"


def test_paused_scheme_blocks_auto_decision(db_session):
    db_session.add(SchemePause(scheme="Personal Loan", reason="test pause", gap_pp=12.0))
    db_session.commit()
    record = run_assessment(db_session, _complete_app(db_session))
    assert record.recommendation == "Refer"
    assert record.escalation_reason_code == "scheme_paused_fairness"


def test_escalated_decisions_appear_in_reasoning(db_session, monkeypatch):
    monkeypatch.setattr(assessment_module, "score_application", lambda profile: (0.52, "Medium"))
    record = run_assessment(db_session, _complete_app(db_session))
    assert record.escalation_flag is True
    assert record.underwriter_action is None  # would surface in the human-review queue
