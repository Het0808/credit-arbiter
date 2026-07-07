from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Application, User
from ..auth import get_current_user
from ..schemas import ApplicationDetail, ApplicationIngestRequest, ApplicationSummary
from ..services.ingestion import ingest_row

router = APIRouter(prefix="/applications", tags=["applications"])


@router.get("", response_model=list[ApplicationSummary])
def list_applications(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(Application).order_by(Application.created_at.desc()).all()


@router.get("/{application_id}", response_model=ApplicationDetail)
def get_application(
    application_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    application = db.query(Application).filter(Application.id == application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    return application


@router.post("/ingest", response_model=ApplicationDetail)
def ingest_application(
    payload: ApplicationIngestRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = payload.model_dump()
    try:
        return ingest_row(db, row)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
