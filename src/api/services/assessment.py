"""Recommendation generation (FR-8 / US-106): composes risk score, retrieved
policy clause, and regulatory verification into an Approve/Decline/Refer
recommendation with an evidence chain, persisted as a decision_record row
(FR-10 / US-108).

This is a Sprint-1 POC-level composition, not FR-3's full multi-scheme policy
rule engine (which is out of scope until Sprint 2). All three sub-services
are called in-process (not over HTTP), so the whole assessment stays well
inside the end-to-end latency budget.
"""

import json

from sqlalchemy.orm import Session

from ..models import Application, DecisionRecord
from .regulatory import verify_regulatory
from .retrieval import retrieve_for_profile
from .scoring import score_application

# regulatory_status + risk_band -> recommendation. Anything not covered here
# (or reached via the kill-switch below) forces a Refer.
_RECOMMENDATION_RULES = {
    ("PASS", "Low"): "Approve",
    ("PASS", "Medium"): "Refer",
    ("PASS", "High"): "Decline",
    ("FAIL", "Low"): "Decline",
    ("FAIL", "Medium"): "Decline",
    ("FAIL", "High"): "Decline",
}

_ESCALATING_RECOMMENDATIONS = {"Refer"}


def _profile_from_application(application: Application) -> dict:
    return {
        "amt_income_total": application.amt_income_total,
        "amt_credit": application.amt_credit,
        "amt_annuity": application.amt_annuity,
        "days_employed": application.days_employed,
        "region_rating_client": application.region_rating_client,
    }


def run_assessment(db: Session, application: Application, force_regulatory_fail: bool = False) -> DecisionRecord:
    profile = _profile_from_application(application)

    risk_score = None
    risk_band = None
    try:
        risk_score, risk_band = score_application(profile)
    except (ValueError, TypeError):
        pass  # missing/invalid inputs -> kill-switch below forces Refer

    retrieval = retrieve_for_profile(profile)
    top_clause = retrieval["clauses"][0] if retrieval["clauses"] else None

    regulatory = verify_regulatory(application.external_id, force_fail=force_regulatory_fail)

    kill_switch_reason = None
    if risk_score is None:
        kill_switch_reason = "missing_risk_score"
    elif retrieval["retrieval_failed"]:
        kill_switch_reason = "retrieval_failed"

    if kill_switch_reason:
        recommendation = "Refer"
        escalation_flag = True
    elif regulatory["status"] == "escalate_for_review":
        recommendation = "Refer"
        escalation_flag = True
    else:
        recommendation = _RECOMMENDATION_RULES.get((regulatory["status"], risk_band), "Refer")
        escalation_flag = recommendation in _ESCALATING_RECOMMENDATIONS

    evidence_chain = {
        "risk_score": risk_score,
        "risk_band": risk_band,
        "retrieved_clause_id": top_clause["clause_id"] if top_clause else None,
        "retrieval_confidence": top_clause["score"] if top_clause else None,
        "regulatory_status": regulatory["status"],
        "regulatory_reason": regulatory["reason"],
        "rule_applied": f"{regulatory['status']}+{risk_band}" if not kill_switch_reason else "kill_switch",
        "kill_switch_reason": kill_switch_reason,
    }

    record = DecisionRecord(
        application_id=application.id,
        risk_score=risk_score,
        risk_band=risk_band,
        retrieved_clause_id=top_clause["clause_id"] if top_clause else None,
        retrieved_clause_text=top_clause["text"] if top_clause else None,
        retrieval_confidence=top_clause["score"] if top_clause else None,
        retrieval_failed=retrieval["retrieval_failed"],
        regulatory_status=regulatory["status"],
        recommendation=recommendation,
        evidence_chain_json=json.dumps(evidence_chain),
        escalation_flag=escalation_flag,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
