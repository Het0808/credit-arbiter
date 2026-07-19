from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import User
from ..services import kill_switch
from ..services.audit_log import reconstruct, verify_chain
from ..services.ops_metrics import compute_dashboard

router = APIRouter(prefix="/ops", tags=["ops"])


class KillSwitchRequest(BaseModel):
    active: bool


@router.get("/dashboard")
def dashboard(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """PRD §8 operations KPIs with thresholds + alerts (US-407)."""
    return compute_dashboard(db)


@router.get("/kill-switch")
def kill_switch_status(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return kill_switch.status(db)


@router.post("/kill-switch")
def set_kill_switch(
    request: KillSwitchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Operator kill-switch: when active, all new assessments route to humans (US-405)."""
    return kill_switch.set_kill_switch(db, request.active, actor=current_user.email)


@router.get("/audit/verify")
def audit_verify(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Verify the immutable audit chain integrity (US-401)."""
    return verify_chain(db)


@router.get("/audit/reconstruct/{decision_record_id}")
def audit_reconstruct(
    decision_record_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    events = reconstruct(db, decision_record_id)
    if not events:
        raise HTTPException(status_code=404, detail="No audit events for that decision")
    return events
