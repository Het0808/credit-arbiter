from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


# --- Auth (unchanged from the original main.py) ---


class UserCreate(BaseModel):
    email: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


# --- Applications (US-102) ---


class ApplicationIngestRequest(BaseModel):
    """Accepts one CSV-row-shaped JSON object using the original HC2018 column
    names, so it can be passed straight to services.ingestion.normalize_row."""

    model_config = ConfigDict(extra="allow")

    SK_ID_CURR: Optional[str] = None
    NAME_CONTRACT_TYPE: Optional[str] = None
    AMT_INCOME_TOTAL: Optional[str] = None
    AMT_CREDIT: Optional[str] = None
    AMT_ANNUITY: Optional[str] = None
    DAYS_EMPLOYED: Optional[str] = None
    DAYS_BIRTH: Optional[str] = None
    CODE_GENDER: Optional[str] = None
    NAME_EDUCATION_TYPE: Optional[str] = None
    NAME_FAMILY_STATUS: Optional[str] = None
    REGION_RATING_CLIENT: Optional[str] = None
    OCCUPATION_TYPE: Optional[str] = None


class ApplicationSummary(BaseModel):
    id: int
    external_id: str
    name_contract_type: Optional[str]
    amt_income_total: Optional[float]
    amt_credit: Optional[float]
    status: str

    model_config = ConfigDict(from_attributes=True)


class ApplicationDetail(ApplicationSummary):
    amt_annuity: Optional[float]
    days_employed: Optional[int]
    name_education_type: Optional[str]
    name_family_status: Optional[str]
    region_rating_client: Optional[int]
    occupation_type: Optional[str]
    missing_fields: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# --- Scoring (US-103) ---


class ScoreRequest(BaseModel):
    application_id: int


class ScoreResponse(BaseModel):
    probability: float
    band: str


# --- Policy retrieval (US-105) ---


class ClauseMatch(BaseModel):
    clause_id: str
    title: str
    text: str
    score: float


class PolicyRetrieveRequest(BaseModel):
    application_id: Optional[int] = None
    query: Optional[str] = None


class PolicyRetrieveResponse(BaseModel):
    clauses: list[ClauseMatch]
    retrieval_failed: bool


# --- Regulatory stub (US-109) ---


class RegulatoryVerifyRequest(BaseModel):
    application_id: str
    force_fail: bool = False


class RegulatoryVerifyResponse(BaseModel):
    status: str
    reason: Optional[str] = None


# --- Assessment / recommendation (US-106, US-108) ---


class AssessRequest(BaseModel):
    application_id: int
    force_regulatory_fail: bool = False


class DecisionRecordOut(BaseModel):
    id: int
    application_id: int
    risk_score: Optional[float]
    risk_band: Optional[str]
    retrieved_clause_id: Optional[str]
    retrieved_clause_text: Optional[str]
    retrieval_confidence: Optional[float]
    retrieval_failed: bool
    regulatory_status: Optional[str]
    recommendation: str
    evidence_chain: dict
    escalation_flag: bool
    created_at: datetime
    underwriter_action: Optional[str]
    underwriter_reason: Optional[str]
    underwriter_action_at: Optional[datetime]


class DecisionRequest(BaseModel):
    action: Literal["accept", "override"]
    reason: Optional[str] = None
