from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import User
from ..services.fairness_monitor import paused_schemes, release_scheme, run_fairness_monitor

router = APIRouter(prefix="/fairness", tags=["fairness"])


@router.post("/monitor")
def run_monitor(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Run the decision-level fairness audit; hard-blocks (pauses) breaching schemes (US-304)."""
    return run_fairness_monitor(db)


@router.get("/paused-schemes")
def list_paused_schemes(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return [
        {"scheme": p.scheme, "reason": p.reason, "gap_pp": p.gap_pp, "created_at": p.created_at}
        for p in paused_schemes(db)
    ]


@router.post("/release/{scheme}")
def release(scheme: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    released = release_scheme(db, scheme)
    return {"scheme": scheme, "released_pauses": released}


@router.get("/proxy-leakage")
def proxy_leakage(current_user: User = Depends(get_current_user)):
    """Report proxy-feature leakage vs protected attributes (US-305)."""
    from src.risk_model.proxy_leakage import run_proxy_leakage_assessment

    return run_proxy_leakage_assessment()
