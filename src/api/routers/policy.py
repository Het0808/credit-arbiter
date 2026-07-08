from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import Application, User
from ..schemas import PolicyRetrieveRequest, PolicyRetrieveResponse
from ..services.retrieval import retrieve, retrieve_for_profile

router = APIRouter(prefix="/policy", tags=["policy"])


@router.post("/retrieve", response_model=PolicyRetrieveResponse)
def retrieve_policy(
    request: PolicyRetrieveRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if request.query:
        return retrieve(request.query, scheme=request.scheme)

    if request.application_id is not None:
        application = db.query(Application).filter(Application.id == request.application_id).first()
        if not application:
            raise HTTPException(status_code=404, detail="Application not found")
        profile = {
            "amt_credit": application.amt_credit,
            "amt_income_total": application.amt_income_total,
            "amt_annuity": application.amt_annuity,
            "days_employed": application.days_employed,
            "region_rating_client": application.region_rating_client,
        }
        scheme = request.scheme or application.loan_scheme
        return retrieve_for_profile(profile, scheme=scheme)

    raise HTTPException(status_code=400, detail="Either application_id or query is required")
