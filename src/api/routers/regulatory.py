from fastapi import APIRouter, Depends

from ..auth import get_current_user
from ..models import User
from ..schemas import RegulatoryVerifyRequest, RegulatoryVerifyResponse
from ..services.regulatory import verify_regulatory

router = APIRouter(prefix="/regulatory", tags=["regulatory"])


@router.post("/verify", response_model=RegulatoryVerifyResponse)
def verify(request: RegulatoryVerifyRequest, current_user: User = Depends(get_current_user)):
    return verify_regulatory(request.application_id, force_fail=request.force_fail)
