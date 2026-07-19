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
    underwriter_reason_code = Column(String, nullable=True)  # structured reason code (US-308)
    underwriter_action_at = Column(DateTime, nullable=True)
    underwriter_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Escalation routing (US-306) & evidence completeness (US-307)
    escalation_reason_code = Column(String, nullable=True)
    evidence_complete = Column(Boolean, default=False)

    # Cost metering (US-402)
    estimated_cost_usd = Column(Float, nullable=True)
    cost_guardrail_breached = Column(Boolean, default=False)


class AuditEvent(Base):
    """Append-only, hash-chained audit record (FR-10 / US-401).

    Each row stores an event (a decision or an external call) plus a SHA-256
    entry_hash computed over its content and the previous row's hash, forming a
    tamper-evident chain: altering any historical row breaks every hash after
    it. Rows are never updated or deleted.
    """

    __tablename__ = "audit_event"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String, nullable=False, index=True)  # decision | external_call
    application_id = Column(Integer, nullable=True, index=True)
    decision_record_id = Column(Integer, nullable=True, index=True)
    payload_json = Column(Text, nullable=False)
    prev_hash = Column(String, nullable=True)
    entry_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class SystemFlag(Base):
    """Key/value operational state, e.g. the global kill-switch (US-405)."""

    __tablename__ = "system_flag"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, nullable=False, index=True)
    value = Column(String, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Document(Base):
    """An uploaded supporting document linked to an application (FR-5 / US-301).

    Real OCR/extraction is out of scope for the POC; a document optionally
    carries structured extracted fields (declared_name, declared_income) so the
    verification service (US-302) can run completeness + consistency checks.
    """

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("applications.id"), nullable=False, index=True)
    doc_type = Column(String, nullable=False)  # e.g. salary_slip, bank_statement, id_proof
    filename = Column(String, nullable=False)
    content_type = Column(String, nullable=True)
    size_bytes = Column(Integer, nullable=True)
    storage_path = Column(String, nullable=True)
    declared_name = Column(String, nullable=True)
    declared_income = Column(Float, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)


class SchemePause(Base):
    """A loan scheme paused by the fairness hard-block (FR-7 / US-304).

    While an active (released_at IS NULL) row exists for a scheme, the
    assessment service must not auto-decide applications for that scheme.
    """

    __tablename__ = "scheme_pause"

    id = Column(Integer, primary_key=True, index=True)
    scheme = Column(String, nullable=False, index=True)
    reason = Column(Text, nullable=False)
    gap_pp = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    released_at = Column(DateTime, nullable=True)


class PolicyCorpusVersion(Base):
    """Metadata row tracking which policy corpus JSON version is loaded (US-104)."""

    __tablename__ = "policy_corpus_version"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, nullable=False)
    effective_date = Column(String, nullable=False)
    source_file = Column(String, nullable=False)
    clause_count = Column(Integer, nullable=False)
    loaded_at = Column(DateTime, default=datetime.utcnow)
