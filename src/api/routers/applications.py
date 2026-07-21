from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import UNDERWRITER, get_current_user
from ..database import get_db
from ..models import Application, DecisionRecord, User
from ..schemas import ApplicationDetail, ApplicationIngestRequest, ApplicationStatus, ApplicationSummary
from ..services.ingestion import ingest_row

router = APIRouter(prefix="/applications", tags=["applications"])


def _is_ops(user: User) -> bool:
    return user.role == UNDERWRITER


def _decision_status(db: Session, application_id: int) -> str:
    """Derive the applicant-facing status from the latest decision.

    Pending  = not yet assessed, or assessed but not actioned, or referred.
    Approved = underwriter accepted an Approve (or overrode a Decline).
    Denied   = underwriter accepted a Decline (or overrode an Approve).
    """
    rec = (
        db.query(DecisionRecord)
        .filter(DecisionRecord.application_id == application_id)
        .order_by(DecisionRecord.created_at.desc())
        .first()
    )
    if not rec or rec.underwriter_action is None:
        return "Pending"
    if rec.underwriter_action == "accept":
        return {"Approve": "Approved", "Decline": "Denied"}.get(rec.recommendation, "Pending")
    # override reverses the system recommendation
    return {"Approve": "Denied", "Decline": "Approved"}.get(rec.recommendation, "Pending")


@router.get("", response_model=list[ApplicationSummary])
def list_applications(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Ops see every application; applicants see only their own."""
    query = db.query(Application)
    if not _is_ops(current_user):
        query = query.filter(Application.owner_id == current_user.id)
    return query.order_by(Application.created_at.desc()).all()


@router.get("/my", response_model=list[ApplicationStatus])
def my_applications(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """The current user's own applications with their decision status."""
    apps = (
        db.query(Application)
        .filter(Application.owner_id == current_user.id)
        .order_by(Application.created_at.desc())
        .all()
    )
    return [
        ApplicationStatus(
            id=a.id, external_id=a.external_id, loan_scheme=a.loan_scheme,
            amt_credit=a.amt_credit, amt_income_total=a.amt_income_total,
            status=a.status, decision_status=_decision_status(db, a.id), created_at=a.created_at,
        )
        for a in apps
    ]


@router.get("/{application_id}", response_model=ApplicationDetail)
def get_application(
    application_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    application = db.query(Application).filter(Application.id == application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    if not _is_ops(current_user) and application.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="You can only view your own applications")
    return application


@router.post("/ingest", response_model=ApplicationDetail)
def ingest_application(
    payload: ApplicationIngestRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = payload.model_dump()
    external_id = (row.get("SK_ID_CURR") or "").strip()

    # Applicants cannot overwrite an application that belongs to someone else.
    if external_id:
        existing = db.query(Application).filter(Application.external_id == external_id).first()
        if existing and existing.owner_id and existing.owner_id != current_user.id and not _is_ops(current_user):
            raise HTTPException(status_code=403, detail="This application ID belongs to another user")

    try:
        application = ingest_row(db, row)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if application.owner_id is None:
        application.owner_id = current_user.id
        db.commit()
        db.refresh(application)
    return application
