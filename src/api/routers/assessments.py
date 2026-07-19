import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import Application, DecisionRecord, User
from ..schemas import AssessRequest, DecisionRecordOut, DecisionRequest
from ..services.assessment import run_assessment

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
        regulatory_status=record.regulatory_status,
        recommendation=record.recommendation,
        evidence_chain=json.loads(record.evidence_chain_json),
        escalation_flag=record.escalation_flag,
        created_at=record.created_at,
        underwriter_action=record.underwriter_action,
        underwriter_reason=record.underwriter_reason,
        underwriter_reason_code=record.underwriter_reason_code,
        underwriter_action_at=record.underwriter_action_at,
        cost_usd=record.cost_usd,
        record_hash=record.record_hash,
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


@router.get("/{assessment_id}", response_model=DecisionRecordOut)
def get_assessment(
    assessment_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    record = db.query(DecisionRecord).filter(DecisionRecord.id == assessment_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return _to_decision_record_out(record)


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

    if request.action == "override" and not request.reason:
        raise HTTPException(status_code=400, detail="A reason is required to override a recommendation")

    record.underwriter_action = request.action
    record.underwriter_reason = request.reason
    record.underwriter_reason_code = request.reason_code
    record.underwriter_action_at = datetime.utcnow()
    record.underwriter_user_id = current_user.id
    db.commit()
    db.refresh(record)
    return _to_decision_record_out(record)
