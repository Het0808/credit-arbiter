"""Tests for audit log, cost metering, and kill-switch (US-401, US-402, US-405)."""

import json

from src.api.models import Application
from src.api.services import kill_switch
from src.api.services.assessment import run_assessment
from src.api.services.audit_log import append_event, reconstruct, verify_chain
from src.api.services.cost_meter import estimate_cost
from src.api.models import AuditEvent


def _app(db):
    a = Application(external_id="g-1", loan_scheme="Personal Loan", amt_income_total=250000,
                    amt_credit=300000, amt_annuity=15000, days_employed=-3650, status="COMPLETE")
    db.add(a); db.commit(); db.refresh(a)
    return a


# --- US-401 audit chain ---

def test_audit_chain_is_intact_and_reconstructable(db_session):
    run_assessment(db_session, _app(db_session))
    result = verify_chain(db_session)
    assert result["intact"] is True
    assert result["event_count"] >= 2  # external_call + decision


def test_tampering_breaks_the_chain(db_session):
    append_event(db_session, "decision", {"a": 1})
    append_event(db_session, "decision", {"a": 2})
    # Tamper with the first row's payload.
    first = db_session.query(AuditEvent).order_by(AuditEvent.id.asc()).first()
    first.payload_json = json.dumps({"a": 999})
    db_session.commit()
    result = verify_chain(db_session)
    assert result["intact"] is False
    assert result["broken_at_id"] == first.id


def test_reconstruct_returns_decision_artifacts(db_session):
    record = run_assessment(db_session, _app(db_session))
    events = reconstruct(db_session, record.id)
    types = {e["event_type"] for e in events}
    assert "decision" in types and "external_call" in types


def test_concurrent_appends_keep_chain_intact(tmp_path):
    # Parallel appends must not fork the hash chain (the _APPEND_LOCK serialises them).
    from concurrent.futures import ThreadPoolExecutor

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from src.api.database import Base

    engine = create_engine(
        f"sqlite:///{tmp_path/'audit.db'}", connect_args={"check_same_thread": False, "timeout": 30}
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    def worker(i):
        s = Session()
        try:
            append_event(s, "decision", {"i": i})
        finally:
            s.close()

    with ThreadPoolExecutor(max_workers=20) as ex:
        list(ex.map(worker, range(60)))

    checker = Session()
    try:
        result = verify_chain(checker)
    finally:
        checker.close()
    assert result["intact"] is True
    assert result["event_count"] == 60


# --- US-402 cost metering ---

def test_normal_cost_within_budget():
    cost = estimate_cost(num_retrieved_clauses=3, num_regulatory_checks=4, projected_explanation_tokens=500)
    assert cost["breached"] is False
    assert cost["cost_usd"] <= 0.08


def test_high_projected_tokens_breach_cost_ceiling():
    cost = estimate_cost(num_retrieved_clauses=3, num_regulatory_checks=4, projected_explanation_tokens=5000)
    assert cost["breached"] is True


def test_assessment_persists_cost(db_session):
    record = run_assessment(db_session, _app(db_session))
    assert record.estimated_cost_usd is not None
    assert record.estimated_cost_usd > 0


# --- US-405 kill-switch ---

def test_kill_switch_routes_to_human(db_session):
    kill_switch.set_kill_switch(db_session, True, actor="ops")
    assert kill_switch.is_active(db_session) is True
    record = run_assessment(db_session, _app(db_session))
    assert record.recommendation == "Refer"
    assert record.escalation_reason_code == "kill_switch_active"


def test_kill_switch_off_allows_normal_flow(db_session):
    kill_switch.set_kill_switch(db_session, False)
    record = run_assessment(db_session, _app(db_session))
    assert record.escalation_reason_code != "kill_switch_active"
