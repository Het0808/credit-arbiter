import json

import pytest

from src.api.services import assessment as assessment_module
from src.api.services.assessment import run_assessment


def test_normal_path_low_risk_pass_regulatory_yields_approve(db_session, complete_application):
    record = run_assessment(db_session, complete_application)
    assert record.recommendation in {"Approve", "Refer", "Decline"}
    assert record.risk_score is not None
    assert record.retrieval_failed is False
    evidence = json.loads(record.evidence_chain_json)
    assert evidence["kill_switch_reason"] is None
    assert evidence["regulatory_status"] in {"PASS", "FAIL"}


def test_missing_risk_score_forces_refer_kill_switch(db_session, incomplete_application):
    record = run_assessment(db_session, incomplete_application)
    assert record.recommendation == "Refer"
    assert record.escalation_flag is True
    evidence = json.loads(record.evidence_chain_json)
    assert evidence["kill_switch_reason"] == "missing_risk_score"


def test_retrieval_failure_forces_refer_kill_switch(db_session, complete_application, monkeypatch):
    def _fake_retrieve_for_profile(profile, top_k=3):
        return {"clauses": [], "retrieval_failed": True}

    monkeypatch.setattr(assessment_module, "retrieve_for_profile", _fake_retrieve_for_profile)

    record = run_assessment(db_session, complete_application)
    assert record.recommendation == "Refer"
    assert record.escalation_flag is True
    assert record.retrieval_failed is True
    evidence = json.loads(record.evidence_chain_json)
    assert evidence["kill_switch_reason"] == "retrieval_failed"


def test_forced_regulatory_failure_escalates_to_refer(db_session, complete_application):
    record = run_assessment(db_session, complete_application, force_regulatory_fail=True)
    assert record.recommendation == "Refer"
    assert record.escalation_flag is True
    assert record.regulatory_status == "escalate_for_review"


@pytest.mark.parametrize(
    "regulatory_status,risk_band,expected_recommendation",
    [
        ("PASS", "Low", "Approve"),
        ("PASS", "Medium", "Refer"),
        ("PASS", "High", "Decline"),
        ("FAIL", "Low", "Decline"),
        ("FAIL", "High", "Decline"),
    ],
)
def test_recommendation_rule_table(
    db_session, complete_application, monkeypatch, regulatory_status, risk_band, expected_recommendation
):
    monkeypatch.setattr(assessment_module, "score_application", lambda profile: (0.5, risk_band))
    monkeypatch.setattr(
        assessment_module,
        "verify_regulatory",
        lambda application_id, force_fail=False: {"status": regulatory_status, "reason": None},
    )

    record = run_assessment(db_session, complete_application)
    assert record.recommendation == expected_recommendation

    should_escalate = expected_recommendation == "Refer"
    assert record.escalation_flag is should_escalate


def test_persists_exactly_one_decision_record_per_call(db_session, complete_application):
    from src.api.models import DecisionRecord

    run_assessment(db_session, complete_application)
    run_assessment(db_session, complete_application)
    count = db_session.query(DecisionRecord).filter(
        DecisionRecord.application_id == complete_application.id
    ).count()
    assert count == 2
