"""Recommendation generation (FR-8), hardened through Sprint 3.

Composes every evidence component into an Approve/Decline/Refer recommendation
with a complete, auditable evidence chain, persisted as a decision_record row.

Layers enforced here:
  - Policy adherence (US-206): a failed policy rule can never yield Approve.
  - Evidence completeness (US-307): the 6 evidence components (risk, risk-factors,
    policy clauses, document findings, regulatory result, fairness result) must all
    be present or the recommendation is Refer (kill-switch).
  - Escalation (US-306): low model confidence (<0.60), a fairness scheme pause, a
    failed/at-risk regulatory verdict, or any policy escalation routes the case to
    the human-review queue with a reason code, and blocks an auto-Approve.
"""

import json

from sqlalchemy.orm import Session

from ..models import Application, DecisionRecord
from .audit_log import append_event
from .cost_meter import estimate_cost, estimate_explanation_tokens
from .document_service import verify_documents
from .fairness_monitor import is_scheme_paused
from .kill_switch import is_active as kill_switch_active
from .policy_engine import evaluate as evaluate_policy
from .regulatory import verify_regulatory
from .retrieval import retrieve_for_profile
from .scoring import explain_score, score_application

DEFAULT_SCHEME = "Personal Loan"
MIN_MODEL_CONFIDENCE = 0.60  # US-306: below this, force Refer

_RECOMMENDATION_RULES = {
    ("PASS", "Low"): "Approve",
    ("PASS", "Medium"): "Refer",
    ("PASS", "High"): "Decline",
    ("FAIL", "Low"): "Decline",
    ("FAIL", "Medium"): "Decline",
    ("FAIL", "High"): "Decline",
}

_ESCALATING_RECOMMENDATIONS = {"Refer"}
_POLICY_ACTION_TO_RECOMMENDATION = {"decline": "Decline", "refer": "Refer", "escalate": "Refer"}

# The six evidence components required for an auditable decision (US-307).
_EVIDENCE_COMPONENTS = ["risk_score", "risk_factors", "policy_clauses", "document_findings",
                        "regulatory_result", "fairness_result"]


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
    scheme = application.loan_scheme or DEFAULT_SCHEME

    # --- Risk (component 1) + risk factors (component 2) ---
    risk_score = risk_band = None
    try:
        risk_score, risk_band = score_application(profile)
    except (ValueError, TypeError):
        pass
    risk_factors = explain_score(profile)
    model_confidence = max(risk_score, 1 - risk_score) if risk_score is not None else None

    # --- Policy clauses (component 3) ---
    retrieval = retrieve_for_profile(profile, scheme=scheme)
    clauses = retrieval["clauses"]
    top_clause = clauses[0] if clauses else None
    corpus_version = retrieval.get("corpus_version")
    policy = evaluate_policy(clauses, profile, scheme=scheme, corpus_version=corpus_version)

    # --- Document findings (component 4) ---
    doc_report = verify_documents(db, application)

    # --- Regulatory result (component 5) ---
    regulatory = verify_regulatory(application.external_id, force_fail=force_regulatory_fail)

    # --- Fairness result (component 6) ---
    scheme_paused = is_scheme_paused(db, scheme)
    fairness_result = {"scheme_paused": scheme_paused, "scheme": scheme}

    # --- Cost metering (US-402) & kill-switch (US-405) ---
    cost = estimate_cost(
        num_retrieved_clauses=len(clauses),
        num_regulatory_checks=len(regulatory.get("checks") or []),
        ran_model_inference=risk_score is not None,
        ran_document_check=True,
        projected_explanation_tokens=estimate_explanation_tokens(len(clauses), len(risk_factors)),
    )
    kill_switch_on = kill_switch_active(db)

    # --- Evidence completeness (US-307) ---
    evidence_present = {
        "risk_score": risk_score is not None,
        "risk_factors": bool(risk_factors),
        "policy_clauses": bool(clauses),
        "document_findings": doc_report is not None,
        "regulatory_result": regulatory.get("status") is not None,
        "fairness_result": fairness_result is not None,
    }
    missing_components = [c for c in _EVIDENCE_COMPONENTS if not evidence_present[c]]
    evidence_complete = not missing_components

    # --- Decision + escalation logic ---
    escalation_reason_code = None
    if kill_switch_on:
        recommendation, escalation_flag = "Refer", True
        escalation_reason_code = "kill_switch_active"
        kill_switch_reason = "kill_switch_active"
    elif cost["breached"]:
        recommendation, escalation_flag = "Refer", True
        escalation_reason_code = "cost_guardrail"
        kill_switch_reason = "cost_guardrail_exceeded"
    elif retrieval["retrieval_failed"]:
        recommendation, escalation_flag = "Refer", True
        escalation_reason_code = "retrieval_failed"
        kill_switch_reason = "retrieval_failed"
    elif not evidence_complete:
        recommendation, escalation_flag = "Refer", True
        escalation_reason_code = "incomplete_evidence"
        kill_switch_reason = f"missing_evidence: {','.join(missing_components)}"
    elif scheme_paused:
        recommendation, escalation_flag = "Refer", True
        escalation_reason_code = "scheme_paused_fairness"
        kill_switch_reason = "scheme_paused_fairness"
    elif model_confidence is not None and model_confidence < MIN_MODEL_CONFIDENCE:
        recommendation, escalation_flag = "Refer", True
        escalation_reason_code = "low_model_confidence"
        kill_switch_reason = None
    elif regulatory["status"] == "escalate_for_review":
        recommendation, escalation_flag = "Refer", True
        escalation_reason_code = "regulatory_unresolved"
        kill_switch_reason = None
    else:
        kill_switch_reason = None
        recommendation = _RECOMMENDATION_RULES.get((regulatory["status"], risk_band), "Refer")
        escalation_flag = recommendation in _ESCALATING_RECOMMENDATIONS

        if not policy["approve_allowed"] and recommendation == "Approve":
            recommendation = _POLICY_ACTION_TO_RECOMMENDATION.get(policy["required_action"], "Refer")
        if not doc_report["verified"] and recommendation == "Approve":
            recommendation = "Refer"  # incomplete/inconsistent docs cannot auto-approve
            escalation_reason_code = "document_findings"
        if policy["escalation_required"]:
            escalation_flag = True
            escalation_reason_code = escalation_reason_code or "policy_escalation"
        escalation_flag = escalation_flag or recommendation in _ESCALATING_RECOMMENDATIONS
        if escalation_flag and escalation_reason_code is None and recommendation == "Refer":
            escalation_reason_code = "refer_recommendation"

    evidence_chain = {
        "risk_score": risk_score,
        "risk_band": risk_band,
        "model_confidence": round(model_confidence, 4) if model_confidence is not None else None,
        "risk_factors": risk_factors,
        "loan_scheme": scheme,
        "policy_version": corpus_version,
        "retrieved_clause_id": top_clause["clause_id"] if top_clause else None,
        "retrieved_source_id": top_clause.get("source_id") if top_clause else None,
        "retrieval_confidence": top_clause["score"] if top_clause else None,
        "policy_clauses": [
            {"clause_id": c["clause_id"], "source_id": c.get("source_id"), "corpus_version": c.get("corpus_version")}
            for c in clauses
        ],
        "policy_passed_rules": policy["passed_rules"],
        "policy_failed_rules": policy["failed_rules"],
        "policy_adherence": policy["policy_adherence"],
        "policy_required_action": policy["required_action"],
        "document_findings": doc_report,
        "regulatory_status": regulatory["status"],
        "regulatory_reason": regulatory["reason"],
        "regulatory_checks": regulatory.get("checks"),
        "fairness_result": fairness_result,
        "estimated_cost_usd": cost["cost_usd"],
        "cost_breakdown": cost["breakdown"],
        "cost_guardrail_breached": cost["breached"],
        "kill_switch_active": kill_switch_on,
        "evidence_present": evidence_present,
        "evidence_complete": evidence_complete,
        "missing_evidence": missing_components,
        "escalation_reason_code": escalation_reason_code,
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
        policy_version=corpus_version,
        loan_scheme=scheme,
        regulatory_status=regulatory["status"],
        recommendation=recommendation,
        evidence_chain_json=json.dumps(evidence_chain),
        escalation_flag=escalation_flag,
        escalation_reason_code=escalation_reason_code,
        evidence_complete=evidence_complete,
        estimated_cost_usd=cost["cost_usd"],
        cost_guardrail_breached=cost["breached"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    # --- Immutable audit trail (US-401): log the external call then the decision ---
    append_event(
        db,
        "external_call",
        {"service": "regulatory", "status": regulatory["status"], "checks": regulatory.get("checks")},
        application_id=application.id,
        decision_record_id=record.id,
    )
    append_event(
        db,
        "decision",
        {"recommendation": recommendation, "evidence_chain": evidence_chain},
        application_id=application.id,
        decision_record_id=record.id,
    )
    return record
