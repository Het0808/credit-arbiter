from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from .database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="underwriter")


class Application(Base):
    """Normalised loan application profile (FR-1 / US-102).

    days_birth and code_gender are stored for fairness-audit purposes only and
    must never be read by src/api/services/scoring.py (assumption A-8b).
    """

    __tablename__ = "applications"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String, unique=True, index=True, nullable=False)
    name_contract_type = Column(String, nullable=True)
    loan_scheme = Column(String, nullable=True, default="Personal Loan")
    amt_income_total = Column(Float, nullable=True)
    amt_credit = Column(Float, nullable=True)
    amt_annuity = Column(Float, nullable=True)
    days_employed = Column(Integer, nullable=True)
    days_birth = Column(Integer, nullable=True)
    code_gender = Column(String, nullable=True)
    name_education_type = Column(String, nullable=True)
    name_family_status = Column(String, nullable=True)
    region_rating_client = Column(Integer, nullable=True)
    occupation_type = Column(String, nullable=True)

    # ML-predictive fields added for the Sprint 2 trained risk model
    # (src/risk_model/). Optional: gracefully imputed by the model pipeline
    # when absent, never required for ingestion completeness.
    ext_source_1 = Column(Float, nullable=True)
    ext_source_2 = Column(Float, nullable=True)
    ext_source_3 = Column(Float, nullable=True)
    cnt_fam_members = Column(Float, nullable=True)
    amt_goods_price = Column(Float, nullable=True)
    cnt_children = Column(Integer, nullable=True)
    flag_own_car = Column(String, nullable=True)
    flag_own_realty = Column(String, nullable=True)
    name_income_type = Column(String, nullable=True)

    loan_scheme = Column(String, nullable=True)

    status = Column(String, default="INCOMPLETE")
    missing_fields = Column(String, nullable=True)
    raw_row_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class DecisionRecord(Base):
    """Immutable-ish audit record for one /assess call (FR-10 / US-108).

    Underwriter fields are updated in place on accept/override rather than
    appended as a separate row - a deliberate Sprint-1 simplification; a
    stricter insert-only audit_event table is deferred to a later sprint.
    """

    __tablename__ = "decision_record"

    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("applications.id"), nullable=False)

    risk_score = Column(Float, nullable=True)
    risk_band = Column(String, nullable=True)

    retrieved_clause_id = Column(String, nullable=True)
    retrieved_clause_text = Column(Text, nullable=True)
    retrieval_confidence = Column(Float, nullable=True)
    retrieval_failed = Column(Boolean, default=False)
    policy_version = Column(String, nullable=True)  # corpus version used (US-207)
    loan_scheme = Column(String, nullable=True)

    regulatory_status = Column(String, nullable=True)

    recommendation = Column(String, nullable=False)
    evidence_chain_json = Column(Text, nullable=False)
    escalation_flag = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    underwriter_action = Column(String, nullable=True)  # accept | override
    underwriter_reason = Column(Text, nullable=True)
    underwriter_reason_code = Column(String, nullable=True)
    underwriter_action_at = Column(DateTime, nullable=True)
    underwriter_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Tamper-evidence hash chain (US-401): sha256(record content + previous
    # record's hash). A POC-appropriate approximation of immutability, not
    # true WORM storage.
    record_hash = Column(String, nullable=True)
    cost_usd = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)


class Document(Base):
    """Uploaded applicant document (FR-5 / US-301)."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("applications.id"), nullable=False)
    doc_type = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)


class PolicyCorpusVersion(Base):
    """Metadata row tracking which policy corpus JSON version is loaded (US-104)."""

    __tablename__ = "policy_corpus_version"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, nullable=False)
    effective_date = Column(String, nullable=False)
    source_file = Column(String, nullable=False)
    clause_count = Column(Integer, nullable=False)
    loaded_at = Column(DateTime, default=datetime.utcnow)
