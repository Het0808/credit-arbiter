from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import Application, PolicyCorpusVersion, User
from ..schemas import PolicyReindexResponse, PolicyRetrieveRequest, PolicyRetrieveResponse
from ..services.retrieval import POLICY_CORPUS_PATH, reindex_corpus, retrieve, retrieve_for_profile

router = APIRouter(prefix="/policy", tags=["policy"])


@router.post("/reindex", response_model=PolicyReindexResponse)
def reindex_policy_corpus(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Manual re-index trigger (US-207): reload the policy corpus from disk
    and make it live immediately, without a server restart. Run this after
    a Compliance SME edits/bumps the version in data/policy_corpus_v1.0.json."""
    metadata = reindex_corpus()

    existing = (
        db.query(PolicyCorpusVersion)
        .filter(
            PolicyCorpusVersion.version == metadata["version"],
            PolicyCorpusVersion.source_file == POLICY_CORPUS_PATH,
        )
        .first()
    )
    if not existing:
        db.add(
            PolicyCorpusVersion(
                version=metadata["version"],
                effective_date=metadata["effective_date"],
                source_file=POLICY_CORPUS_PATH,
                clause_count=metadata["clause_count"],
            )
        )
        db.commit()

    return metadata


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
