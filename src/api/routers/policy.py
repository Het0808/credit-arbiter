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


@router.post("/evaluate", response_model=PolicyEvaluateResponse)
def evaluate_policy_for_application(
    request: PolicyEvaluateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    application = db.query(Application).filter(Application.id == request.application_id).first()
    if not application:
        raise HTTPException(status_code=404, detail="Application not found")
    scheme = request.scheme or application.loan_scheme
    profile = _profile(application)
    retrieval = retrieval_service.retrieve_for_profile(
        profile, scheme=scheme, corpus_version=request.corpus_version
    )
    return evaluate_policy(
        retrieval["clauses"], profile, scheme=scheme, corpus_version=retrieval.get("corpus_version")
    )


@router.get("/versions")
def policy_versions(current_user: User = Depends(get_current_user)):
    """List all available corpus versions and which one is active (US-207)."""
    return retrieval_service.list_versions()


@router.get("/corpus")
def active_corpus(current_user: User = Depends(get_current_user)):
    return retrieval_service.corpus_metadata()


@router.get("/clause/{clause_id}")
def get_clause(
    clause_id: str,
    corpus_version: str | None = None,
    current_user: User = Depends(get_current_user),
):
    """Return a single clause's text + version for citation display (US-309)."""
    clause = retrieval_service.get_clause(clause_id, corpus_version)
    if not clause:
        raise HTTPException(status_code=404, detail="Clause not found")
    return clause


@router.post("/reindex")
def reindex_policy(
    request: PolicyReindexRequest,
    current_user: User = Depends(get_current_user),
):
    """Manual re-index trigger: re-scan data/ and activate a corpus version (US-207)."""
    try:
        return retrieval_service.reindex(request.version)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
