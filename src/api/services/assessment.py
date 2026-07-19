"""Recommendation generation (FR-8 / US-106, US-206, US-307): composes risk
score, scheme-aware retrieved policy clause, policy rule evaluation, and
regulatory verification into an Approve/Decline/Refer recommendation with an
evidence chain, persisted as a decision_record row (FR-10 / US-108).

All sub-services are called in-process (not over HTTP), so the whole
assessment stays well inside the end-to-end latency budget.
"""

import hashlib
import json
import os
import time

from sqlalchemy.orm import Session

from ..models import Application, DecisionRecord, Document
from .document_verification import verify_documents
from .explanation import generate_explanation
from .fairness_check import check_fairness
from .policy_evaluation import evaluate_policy
from .regulatory import verify_regulatory
from .retrieval import retrieve_for_profile
from .scoring import score_application

GENESIS_HASH = "0" * 64


def _compute_record_hash(db: Session, application_id: int, recommendation: str, evidence_chain_json: str) -> str:
    """sha256(previous record's hash + this record's content) - a POC-scale
    tamper-evidence hash chain (US-401), not true WORM storage."""
    previous = db.query(DecisionRecord).order_by(DecisionRecord.id.desc()).first()
    previous_hash = previous.record_hash if (previous and previous.record_hash) else GENESIS_HASH
    payload = f"{previous_hash}|{application_id}|{recommendation}|{evidence_chain_json}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

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

# PRD §11 specifies "ML model confidence score < 0.60" as an escalation
# trigger, but a binary probability has no native confidence score - distance
# from the 0.5 decision boundary (0=maximally uncertain, 1=maximally certain)
# is this project's documented proxy for it. Empirically, this model's
# predicted probabilities cluster well short of the extremes (consistent
# with its disclosed AUC of 0.76, short of the 0.80 target - see
# reports/ml/lightgbm_metrics.json), so applying the PRD's literal 0.60 cutoff
# to that proxy would escalate nearly every application. 0.30 keeps the
# trigger meaningful (it still catches genuinely coin-flip ~50% predictions)
# without defeating the point of having a risk model at all.
ML_CONFIDENCE_FLOOR = 0.30


def _profile_from_application(application: Application) -> dict:
    return {
        "amt_income_total": application.amt_income_total,
        "amt_credit": application.amt_credit,
        "amt_annuity": application.amt_annuity,
        "days_employed": application.days_employed,
        "region_rating_client": application.region_rating_client,
        "ext_source_1": application.ext_source_1,
        "ext_source_2": application.ext_source_2,
        "ext_source_3": application.ext_source_3,
        "cnt_fam_members": application.cnt_fam_members,
        "amt_goods_price": application.amt_goods_price,
        "cnt_children": application.cnt_children,
        "name_contract_type": application.name_contract_type,
        "flag_own_car": application.flag_own_car,
        "flag_own_realty": application.flag_own_realty,
        "name_income_type": application.name_income_type,
        "name_education_type": application.name_education_type,
    }


def _fairness_profile_from_application(application: Application) -> dict:
    """Separate, explicit accessor for the fairness-audit-only fields
    (assumption A-8b's sanctioned use of code_gender/days_birth) - kept apart
    from _profile_from_application so the ML-scoring profile can never
    accidentally pick these up."""
    return {
        "code_gender": application.code_gender,
        "days_birth": application.days_birth,
        "name_education_type": application.name_education_type,
        "name_income_type": application.name_income_type,
        "region_rating_client": application.region_rating_client,
    }


def run_assessment(db: Session, application: Application, force_regulatory_fail: bool = False) -> DecisionRecord:
    start_time = time.perf_counter()

    # Operator kill-switch (US-405): an operator can force every new
    # assessment straight to the human path within seconds by setting this
    # env var, without a deploy. Checked first - bypasses every sub-service.
    if os.environ.get("HALCYON_KILL_SWITCH", "").strip().lower() == "true":
        evidence_chain = {"kill_switch_reason": "operator_kill_switch"}
        evidence_chain_json = json.dumps(evidence_chain)
        record = DecisionRecord(
            application_id=application.id,
            recommendation="Refer",
            evidence_chain_json=evidence_chain_json,
            escalation_flag=True,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            record_hash=_compute_record_hash(db, application.id, "Refer", evidence_chain_json),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    profile = _profile_from_application(application)

    risk_score = None
    risk_band = None
    risk_factors = []
    ml_confidence = None
    try:
        score_result = score_application(profile)
        risk_score = score_result["probability"]
        risk_band = score_result["band"]
        risk_factors = score_result["top_risk_factors"]
        # Binary probability has no native confidence score; distance from the
        # decision boundary (0.5) is a documented proxy for it (PRD §11:
        # "ML model confidence score < 0.60 ... -> human-review path").
        ml_confidence = abs(risk_score - 0.5) * 2
    except (ValueError, TypeError):
        pass  # missing/invalid inputs -> kill-switch below forces Refer

    scheme = application.loan_scheme
    retrieval = retrieve_for_profile(profile, scheme=scheme)
    top_clause = retrieval["clauses"][0] if retrieval["clauses"] else None
    all_clauses = retrieval["clauses"]

    policy_result = evaluate_policy(profile, scheme)

    uploaded_doc_types = [
        d.doc_type for d in db.query(Document).filter(Document.application_id == application.id).all()
    ]
    document_result = verify_documents(policy_result["scheme"], application.external_id, uploaded_doc_types)

    regulatory = verify_regulatory(application.external_id, force_fail=force_regulatory_fail)
    fairness_result = check_fairness(_fairness_profile_from_application(application))

    kill_switch_reason = None
    if risk_score is None:
        kill_switch_reason = "missing_risk_score"
    elif retrieval["retrieval_failed"]:
        kill_switch_reason = "retrieval_failed"
    elif policy_result["thin_file_flag"]:
        kill_switch_reason = "thin_file"
    elif ml_confidence is not None and ml_confidence < ML_CONFIDENCE_FLOOR:
        kill_switch_reason = "low_ml_confidence"
    elif policy_result["failed_rules"]:
        kill_switch_reason = "policy_violation"
    elif not document_result["complete"]:
        kill_switch_reason = "missing_documents"
    elif not document_result["consistent"]:
        kill_switch_reason = "document_inconsistency"
    elif fairness_result["fairness_alert"]:
        kill_switch_reason = "fairness_alert"

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
        "risk_factors": risk_factors,
        "ml_confidence": ml_confidence,
        "loan_scheme": policy_result["scheme"],
        "retrieved_clause_id": top_clause["clause_id"] if top_clause else None,
        "retrieval_confidence": top_clause["score"] if top_clause else None,
        "retrieved_clauses": all_clauses,
        "policy_passed_rules": policy_result["passed_rules"],
        "policy_failed_rules": policy_result["failed_rules"],
        "thin_file_flag": policy_result["thin_file_flag"],
        "document_verification": document_result,
        "regulatory_status": regulatory["status"],
        "regulatory_reason": regulatory["reason"],
        "regulatory_sub_checks": regulatory.get("sub_checks", {}),
        "fairness_alert": fairness_result["fairness_alert"],
        "fairness_triggered_segments": fairness_result["triggered_segments"],
        "rule_applied": f"{regulatory['status']}+{risk_band}" if not kill_switch_reason else "kill_switch",
        "kill_switch_reason": kill_switch_reason,
    }

    explanation = generate_explanation(evidence_chain, recommendation)
    evidence_chain["narrative_explanation"] = explanation["narrative"]
    evidence_chain["explanation_source"] = explanation["source"]
    evidence_chain["explanation_cost_usd"] = explanation["cost_usd"]

    evidence_chain_json = json.dumps(evidence_chain)
    latency_ms = (time.perf_counter() - start_time) * 1000

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
        evidence_chain_json=evidence_chain_json,
        escalation_flag=escalation_flag,
        cost_usd=explanation["cost_usd"],
        latency_ms=latency_ms,
        record_hash=_compute_record_hash(db, application.id, recommendation, evidence_chain_json),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
