from datetime import datetime
from typing import Any, Literal, Optional

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
    EXT_SOURCE_1: Optional[str] = None
    EXT_SOURCE_2: Optional[str] = None
    EXT_SOURCE_3: Optional[str] = None
    CNT_FAM_MEMBERS: Optional[str] = None
    AMT_GOODS_PRICE: Optional[str] = None
    CNT_CHILDREN: Optional[str] = None
    FLAG_OWN_CAR: Optional[str] = None
    FLAG_OWN_REALTY: Optional[str] = None
    NAME_INCOME_TYPE: Optional[str] = None


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
    loan_scheme: Optional[str] = None
    ext_source_1: Optional[float] = None
    ext_source_2: Optional[float] = None
    ext_source_3: Optional[float] = None
    cnt_fam_members: Optional[float] = None
    amt_goods_price: Optional[float] = None
    cnt_children: Optional[int] = None
    flag_own_car: Optional[str] = None
    flag_own_realty: Optional[str] = None
    name_income_type: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# --- Documents (US-301, US-302) ---


class DocumentOut(BaseModel):
    id: int
    application_id: int
    doc_type: str
    filename: str
    uploaded_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentVerificationOut(BaseModel):
    required_documents: list[str]
    missing_documents: list[str]
    complete: bool
    consistent: bool
    consistency_findings: list[str]


# --- Scoring (US-103) ---


class ScoreRequest(BaseModel):
    application_id: int


class RiskFactor(BaseModel):
    feature: str
    label: str
    value: Any
    impact: float


class ScoreResponse(BaseModel):
    probability: float
    band: str
    top_risk_factors: list[RiskFactor] = []


# --- Policy retrieval (US-105) ---


class ClauseMatch(BaseModel):
    clause_id: str
    source_id: str
    scheme: str
    version: str
    title: str
    text: str
    score: float
    corpus_version: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class PolicyRetrieveRequest(BaseModel):
    application_id: Optional[int] = None
    query: Optional[str] = None
    scheme: Optional[str] = None


class PolicyRetrieveResponse(BaseModel):
    clauses: list[ClauseMatch]
    retrieval_failed: bool
    corpus_version: Optional[str] = None
    scheme: Optional[str] = None


# --- Policy evaluation (US-206) & version management (US-207) ---


class PolicyEvaluateRequest(BaseModel):
    application_id: int
    scheme: Optional[str] = None
    corpus_version: Optional[str] = None


class PolicyEvaluateResponse(BaseModel):
    scheme: Optional[str] = None
    corpus_version: Optional[str] = None
    passed_rules: list[str]
    failed_rules: list[dict]
    escalation_required: bool
    required_action: Optional[str] = None
    approve_allowed: bool
    policy_adherence: float
    metrics: dict


class PolicyReindexRequest(BaseModel):
    version: Optional[str] = None


# --- Documents (US-301, US-302) ---


class DocumentOut(BaseModel):
    id: int
    application_id: int
    doc_type: str
    filename: str
    content_type: Optional[str]
    size_bytes: Optional[int]
    declared_name: Optional[str]
    declared_income: Optional[float]
    uploaded_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentVerificationResponse(BaseModel):
    scheme: str
    required_docs: list[str]
    present_docs: list[str]
    missing_information: list[str]
    complete: bool
    consistency_findings: list[dict]
    consistent: bool
    verified: bool
    document_count: int


class PolicyReindexResponse(BaseModel):
    version: str
    effective_date: str
    schemes: list[str]
    clause_count: int


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
    policy_version: Optional[str] = None
    loan_scheme: Optional[str] = None
    regulatory_status: Optional[str]
    recommendation: str
    evidence_chain: dict
    escalation_flag: bool
    escalation_reason_code: Optional[str] = None
    evidence_complete: Optional[bool] = None
    estimated_cost_usd: Optional[float] = None
    cost_guardrail_breached: Optional[bool] = None
    created_at: datetime
    underwriter_action: Optional[str]
    underwriter_reason: Optional[str]
    underwriter_reason_code: Optional[str] = None
    underwriter_action_at: Optional[datetime]
    cost_usd: Optional[float] = None
    record_hash: Optional[str] = None


REASON_CODES = ["insufficient_evidence", "policy_too_strict", "doc_follow_up", "fairness_concern", "other"]


# Structured reason codes an underwriter must pick when overriding (US-308).
OVERRIDE_REASON_CODES = [
    "compensating_factors",
    "policy_exception_approved",
    "additional_documentation",
    "data_quality_issue",
    "risk_appetite",
    "regulatory_clarification",
    "other",
]


class DecisionRequest(BaseModel):
    action: Literal["accept", "override"]
    reason: Optional[str] = None
    reason_code: Optional[
        Literal["insufficient_evidence", "policy_too_strict", "doc_follow_up", "fairness_concern", "other"]
    ] = None
