from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import Application, User
from ..schemas import ScoreRequest, ScoreResponse
from ..services.scoring import score_application

router = APIRouter(prefix="/score", tags=["scoring"])


@router.post("", response_model=ScoreResponse)
def score(
    request: ScoreRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    application = db.query(Application).filter(Application.id == request.application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")

    profile = {
        "amt_income_total": application.amt_income_total,
        "amt_credit": application.amt_credit,
        "amt_annuity": application.amt_annuity,
        "days_employed": application.days_employed,
    }
    try:
        probability, band = score_application(profile)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {"probability": probability, "band": band}
