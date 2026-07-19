import json
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import Application, DecisionRecord, User
from ..schemas import (
    OVERRIDE_REASON_CODES,
    AssessRequest,
    DecisionRecordOut,
    DecisionRequest,
    HumanReviewItem,
)
from ..services.assessment import run_assessment
from ..services.explanation import generate_explanation

router = APIRouter(prefix="/assessments", tags=["assessments"])
assess_router = APIRouter(tags=["assessments"])


def _to_decision_record_out(record: DecisionRecord) -> DecisionRecordOut:
    return DecisionRecordOut(
        id=record.id,
        application_id=record.application_id,
        risk_score=record.risk_score,
        risk_band=record.risk_band,
        retrieved_clause_id=record.retrieved_clause_id,
        retrieved_clause_text=record.retrieved_clause_text,
        retrieval_confidence=record.retrieval_confidence,
        retrieval_failed=record.retrieval_failed,
        policy_version=record.policy_version,
        loan_scheme=record.loan_scheme,
        regulatory_status=record.regulatory_status,
        recommendation=record.recommendation,
        evidence_chain=json.loads(record.evidence_chain_json),
        escalation_flag=record.escalation_flag,
        escalation_reason_code=record.escalation_reason_code,
        evidence_complete=record.evidence_complete,
        estimated_cost_usd=record.estimated_cost_usd,
        cost_guardrail_breached=record.cost_guardrail_breached,
        created_at=record.created_at,
        underwriter_action=record.underwriter_action,
        underwriter_reason=record.underwriter_reason,
        underwriter_reason_code=record.underwriter_reason_code,
        underwriter_action_at=record.underwriter_action_at,
    )


@assess_router.post("/assess", response_model=DecisionRecordOut)
def assess(
    request: AssessRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    application = db.query(Application).filter(Application.id == request.application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")

    record = run_assessment(db, application, force_regulatory_fail=request.force_regulatory_fail)
    return _to_decision_record_out(record)


@router.get("/queue/human-review", response_model=list[HumanReviewItem])
def human_review_queue(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Escalated decisions awaiting human review (FR-11 / US-306).

    An item is any escalated decision (escalation_flag) not yet actioned by an
    underwriter, carrying its escalation reason code.
    """
    records = (
        db.query(DecisionRecord)
        .filter(DecisionRecord.escalation_flag.is_(True), DecisionRecord.underwriter_action.is_(None))
        .order_by(DecisionRecord.created_at.desc())
        .all()
    )
    return [
        HumanReviewItem(
            assessment_id=r.id,
            application_id=r.application_id,
            recommendation=r.recommendation,
            escalation_reason_code=r.escalation_reason_code,
            risk_band=r.risk_band,
            loan_scheme=r.loan_scheme,
            created_at=r.created_at,
        )
        for r in records
    ]


@router.get("/metrics/override-rate")
def override_rate(days: int = 7, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Override-rate metric over the trailing window (US-308)."""
    since = datetime.utcnow() - timedelta(days=days)
    actioned = db.query(DecisionRecord).filter(
        DecisionRecord.underwriter_action.isnot(None), DecisionRecord.underwriter_action_at >= since
    ).all()
    total = len(actioned)
    overrides = sum(1 for r in actioned if r.underwriter_action == "override")
    return {
        "window_days": days,
        "actioned_count": total,
        "override_count": overrides,
        "override_rate": round(overrides / total, 4) if total else 0.0,
    }


@router.get("/{assessment_id}", response_model=DecisionRecordOut)
def get_assessment(
    assessment_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    record = db.query(DecisionRecord).filter(DecisionRecord.id == assessment_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return _to_decision_record_out(record)


@router.get("/{assessment_id}/explanation")
def get_explanation(
    assessment_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    """Grounded narrative explanation assembled from the stored evidence (US-208 stand-in)."""
    record = db.query(DecisionRecord).filter(DecisionRecord.id == assessment_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return generate_explanation(json.loads(record.evidence_chain_json), record.recommendation)


@router.post("/{assessment_id}/decision", response_model=DecisionRecordOut)
def record_decision(
    assessment_id: int,
    request: DecisionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    record = db.query(DecisionRecord).filter(DecisionRecord.id == assessment_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Assessment not found")

    if request.action == "override":
        if not request.reason:
            raise HTTPException(status_code=400, detail="A reason is required to override a recommendation")
        if not request.reason_code:
            raise HTTPException(status_code=400, detail="A reason_code is required to override a recommendation")
        if request.reason_code not in OVERRIDE_REASON_CODES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid reason_code. Allowed: {OVERRIDE_REASON_CODES}",
            )

    record.underwriter_action = request.action
    record.underwriter_reason = request.reason
    record.underwriter_reason_code = request.reason_code if request.action == "override" else None
    record.underwriter_action_at = datetime.utcnow()
    record.underwriter_user_id = current_user.id
    db.commit()
    db.refresh(record)
    return _to_decision_record_out(record)
